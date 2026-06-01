import torch
import torch.nn.functional as F
import copy

from centralized.config import DEVICE


def train_and_eval(model, trainloader, testloader, valloader=None, epochs=10, lr=0.001):
    model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3
    )
    
    best_val_acc = 0.0
    best_state = None
    grad_sim_history = []

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        grad_sim_sum, grad_sim_count = 0.0, 0

        model.train()

        for inputs_dict, labels, is_augmented in trainloader:
            labels = labels.to(DEVICE)
            inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs_dict)

            # Compute total loss for the mixed batch
            loss = criterion(logits, labels)

            aug_mask = is_augmented.bool()
            clean_mask = ~aug_mask

            # Gradient similarity calculations
            # Only compute if both clean and imputed samples in batch
            if clean_mask.any() and aug_mask.any():
                # Get distinct gradients
                loss_clean = criterion(logits[clean_mask], labels[clean_mask])
                #loss_imputed = criterion(logits[aug_mask], labels[aug_mask])

                # Get the parameters of the shared classifier head
                shared_params = list(model.classifier.parameters())

                # Calculate gradients with respect to the shared classifier
                grad_c = torch.autograd.grad(loss_clean, shared_params, retain_graph=True, allow_unused=True)
                grad_total = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
                
                flat_grad_c = torch.cat([g.contiguous().view(-1) for g in grad_c if g is not None])
                flat_grad_total = torch.cat([g.contiguous().view(-1) for g in grad_total if g is not None])

                #grad_i = torch.autograd.grad(loss_imputed, shared_params, retain_graph=True, allow_unused=True)
                #flat_grad_i = torch.cat([g.contiguous().view(-1) for g in grad_i if g is not None])

                #if flat_grad_c.numel() > 0 and flat_grad_i.numel() > 0:
                #    sim = F.cosine_similarity(flat_grad_c, flat_grad_i, dim=0).item()

                if flat_grad_c.numel() > 0 and flat_grad_total.numel() > 0:
                    sim = F.cosine_similarity(flat_grad_c, flat_grad_total, dim=0).item()
                    grad_sim_sum += sim
                    grad_sim_count += 1
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_grad_sim = grad_sim_sum / grad_sim_count if grad_sim_count > 0 else 0.0
        grad_sim_history.append(avg_grad_sim)

        epoch_loss = running_loss / max(1, len(trainloader))
        epoch_acc = correct / total

        # Eval validation set
        if valloader is not None:
            model.eval()
            val_correct, val_total = 0, 0
            val_running_loss = 0.0

            with torch.no_grad():
                for inputs_dict, labels, _ in valloader:
                    labels = labels.to(DEVICE)
                    inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
                    logits = model(inputs_dict)
                    loss = criterion(logits, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / max(1, val_total)
            val_loss = val_running_loss / max(1, len(valloader))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"| Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    # Load best model and evaluate once on test set
    model.load_state_dict(best_state)

    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for inputs_dict, labels, _ in testloader:
            labels = labels.to(DEVICE)
            inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
            logits = model(inputs_dict)
            _, predicted = torch.max(logits.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Calculate final metrics
    final_test_acc = test_correct / test_total
    
    print(f"-> Final Baseline Test Acc (from best val epoch): {final_test_acc:.4f}\n")
    return final_test_acc, grad_sim_history


def train_and_eval_mt(model, trainloader, testloader, valloader=None, epochs=10, lr=0.001, lmbda=0.75):
    """Multi task model trainer and evaluator. Works with 2 output heads."""
    model = model.to(DEVICE)

    # Two separate loss calculations, one for each head
    criterion_primary = torch.nn.CrossEntropyLoss().to(DEVICE)
    criterion_aux = torch.nn.CrossEntropyLoss().to(DEVICE)
    criterion_eval = torch.nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3
    )

    best_val_acc = 0.0
    best_state = None
    grad_sim_history = []

    for epoch in range(epochs):
        running_loss, correct_primary, total_primary = 0.0, 0, 0
        correct_aux, total_aux = 0, 0
        grad_sim_sum, grad_sim_count = 0.0, 0

        model.train()

        for inputs_dict, labels, is_augmented in trainloader:
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            aug_mask = is_augmented.bool()
            clean_mask = ~aug_mask
            loss = None
            loss_primary, loss_aux = None, None

            # STEP 1: only clean data goes through primary head
            if clean_mask.any():
                clean_dict = {k: v[clean_mask].to(DEVICE) for k, v in inputs_dict.items()}
                clean_labels = labels[clean_mask]

                # Forward pass
                logits_primary = model(clean_dict, use_auxiliary=False)
                loss_primary = criterion_primary(logits_primary, clean_labels)

                # Start total loss with weighted primary loss
                loss = loss_primary

                # Metrics for primary head
                _, predicted = torch.max(logits_primary.data, 1)
                total_primary += clean_labels.size(0)
                correct_primary += (predicted == clean_labels).sum().item()

            # STEP 2: only augmented data through auxiliary head
            if aug_mask.any():
                aug_dict = {k: v[aug_mask].to(DEVICE) for k, v in inputs_dict.items()}
                aug_labels = labels[aug_mask]

                # Forward pass
                logits_aux = model(aug_dict, use_auxiliary=True)
                loss_aux = criterion_aux(logits_aux, aug_labels)

                if loss is not None:
                    loss += lmbda * loss_aux
                else:
                    loss = lmbda * loss_aux

                # Metrics
                _, predicted = torch.max(logits_aux.data, 1)
                total_aux += aug_labels.size(0)
                correct_aux += (predicted == aug_labels).sum().item()

            # Gradient similarity calculation
            if loss_primary is not None and loss_aux is not None:
                # Get all shared parameters for gradient similarity calculation
                shared_params = list(model.extractors.parameters())

                # Extract gradients for the shared backbones (extractors) independently
                # Retain graph because loss.backward() is used later on
                grad_p = torch.autograd.grad(
                    loss_primary, shared_params, retain_graph=True, allow_unused=True
                )
                grad_a = torch.autograd.grad(
                    loss_aux, shared_params, retain_graph=True, allow_unused=True
                )

                # Flatten gradients into single 1D vectors, needed for comparison
                flat_grad_p = torch.cat(
                    [g.contiguous().view(-1) for g in grad_p if g is not None]
                )
                flat_grad_a = torch.cat(
                    [g.contiguous().view(-1) for g in grad_a if g is not None]
                )

                # Calculate cosine similarity
                if flat_grad_p.numel() > 0 and flat_grad_a.numel() > 0:
                    sim = F.cosine_similarity(flat_grad_p, flat_grad_a, dim=0).item()
                    grad_sim_sum += sim
                    grad_sim_count += 1
                
            # Backward pass
            if loss is not None:
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        avg_grad_sim = grad_sim_sum / grad_sim_count if grad_sim_count > 0 else 0.0
        grad_sim_history.append(avg_grad_sim)

        acc_primary = correct_primary / max(1, total_primary)
        acc_aux = correct_aux / max(1, total_aux)

        # Evaluate validation set
        if valloader is not None:
            model.eval()
            val_correct, val_total = 0, 0
            val_running_loss = 0.0

            with torch.no_grad():
                for inputs_dict, labels, is_augmented in valloader:
                    labels = labels.to(DEVICE)
                    aug_mask = is_augmented.bool()
                    clean_mask = ~aug_mask
                    batch_logits = torch.zeros(labels.size(0), model.num_classes).to(DEVICE)
                    
                    if clean_mask.any():
                        clean_dict = {k: v[clean_mask].to(DEVICE) for k, v in inputs_dict.items()}
                        batch_logits[clean_mask] = model(clean_dict, use_auxiliary=False)
                    if aug_mask.any():
                        aug_dict = {k: v[aug_mask].to(DEVICE) for k, v in inputs_dict.items()}
                        batch_logits[aug_mask] = model(aug_dict, use_auxiliary=True)

                    # Calculate validation loss
                    loss = criterion_eval(batch_logits, labels)
                    val_running_loss += loss.item()

                    _, predicted = torch.max(batch_logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / max(1, val_total)
            val_loss = val_running_loss / max(1, len(valloader))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | Val Loss: {val_loss:.4f} "
                  f"| Train Clean: {acc_primary:.4f} | Train Aux: {acc_aux:.4f} "
                  f"| Val Acc: {val_acc:.4f} | Grad Sim: {avg_grad_sim:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f} "
                  f"| Train Clean: {acc_primary:.4f} | Train Aux: {acc_aux:.4f} "
                  f"| Grad Sim: {avg_grad_sim:.4f}")

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    # Load best model and evaluate once on test set
    model.load_state_dict(best_state)

    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for inputs_dict, labels, is_augmented in testloader:
            labels = labels.to(DEVICE)

            aug_mask = is_augmented.bool()
            clean_mask = ~aug_mask

            batch_logits = torch.zeros(labels.size(0), model.num_classes).to(DEVICE)

            if clean_mask.any():
                clean_dict = {k: v[clean_mask].to(DEVICE) for k, v in inputs_dict.items()}
                batch_logits[clean_mask] = model(clean_dict, use_auxiliary=False)

            if aug_mask.any():
                aug_dict = {k: v[aug_mask].to(DEVICE) for k, v in inputs_dict.items()}
                batch_logits[aug_mask] = model(aug_dict, use_auxiliary=True)

            _, predicted = torch.max(batch_logits.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    final_test_acc = test_correct / max(1, test_total)

    print(f"-> Final Multi-Task Test Acc (from best val epoch): {final_test_acc:.4f}\n")
    return final_test_acc, grad_sim_history

