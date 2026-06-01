import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader

"""
Pure PyTorch logic. It knows nothing about FL or dataset wrappers. It just takes
a batch of data, pushes it through the NN, calculates the loss, and steps the optimizer.
"""

@dataclass
class TrainConfig:
    """Configuration object for training and testing loop"""
    device: torch.device
    sensor_dims: Dict[int, int]
    num_modalities: int
    num_classes: int = 40
    strategy_name: str = "fedavg"

    # Architecture and imputation
    mt_enabled: bool = False
    mt_alpha: float = 0.0

    # Training specific (defaults to avoid complaining)
    local_epochs: int = 1
    lr: float = 0.001
    momentum: float = 0.0
    weight_decay: float = 0.0

    # Regularization (fedmsplit)
    lambda_reg: float = 0.0
    neighbour_model: Optional[Dict] = None


def train(
        net: Module, trainloader: DataLoader, cfg: TrainConfig
) -> Tuple[float, float]:
    """Train the model on the training set."""
    net.to(cfg.device)
    net.train()

    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    # optimizer = torch.optim.SGD(
    #     net.parameters(),
    #     lr=cfg.lr,
    #     momentum=cfg.momentum,  # 0
    #     weight_decay=cfg.weight_decay  # 0
    # )
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg.lr,
        weight_decay=1e-3
    )

    running_loss, correct, total, n_batches = 0.0, 0, 0, 0

    # Pre-process neighbour parameters for faster access if given (fedmsplit)
    neighbour_params = {}
    if cfg.neighbour_model is not None:
        neighbour_params = {
            k: v.to(cfg.device) for k, v in cfg.neighbour_model.items()
        }

    for _ in range(cfg.local_epochs):
        epoch_cos_sims = []

        for inputs_imputed, labels, is_augmented in trainloader:
            labels = labels.to(cfg.device)
            inputs_imputed = {k: v.to(cfg.device) for k, v in inputs_imputed.items()}
            
            aug_mask = is_augmented.to(cfg.device, dtype=torch.bool)
            clean_mask = ~aug_mask

            # zero gradients before forward pass
            optimizer.zero_grad()
            loss = 0.0

            # 2 head architecture logic
            # Implementation of paper "A good data augmentation policy is not all you need"
            if cfg.mt_enabled:
                # STEP 1: Only clean data goes through primary head
                if clean_mask.sum() > 1:
                    clean_dict = {k: v[clean_mask] for k, v in inputs_imputed.items()}
                    clean_labels = labels[clean_mask]

                    logits_primary = net(clean_dict, use_auxiliary=False)
                    loss_primary = criterion(logits_primary, clean_labels)

                    # Weighted primary loss
                    loss += (1.0 - cfg.mt_alpha) * loss_primary

                    # Metrics for primary head
                    _, predicted = torch.max(logits_primary.data, 1)
                    correct += (predicted == clean_labels).sum().item()
                    total += clean_labels.size(0)

                # STEP 2: Only augmented data goes through auxiliary head
                if aug_mask.sum() > 1:
                    aug_dict = {k: v[aug_mask] for k, v in inputs_imputed.items()}
                    aug_labels = labels[aug_mask]

                    logits_aux = net(aug_dict, use_auxiliary=True)
                    loss_aux = criterion(logits_aux, aug_labels)
                    
                    loss += cfg.mt_alpha * loss_aux

                    # Track metrics
                    _, predicted = torch.max(logits_aux.data, 1)
                    correct += (predicted == aug_labels).sum().item()
                    total += aug_labels.size(0)

            # Standard 1 head architecture
            else:
                # if clean_mask.sum() > 1 and aug_mask.sum() > 1:
                #     # Clean gradients
                #     optimizer.zero_grad()
                #     clean_dict = {k: v[clean_mask] for k, v in inputs_imputed.items()}
                #     loss_c = criterion(net(clean_dict), labels[clean_mask])
                #     loss_c.backward()
                #     # gradients from first layer of network
                #     grad_c = list(net.parameters())[0].grad.clone().flatten()

                #     # Augmented gradients
                #     optimizer.zero_grad()
                #     aug_dict = {k: v[aug_mask] for k, v in inputs_imputed.items()}
                #     loss_a = criterion(net(aug_dict), labels[aug_mask])
                #     loss_a.backward()
                #     grad_a = list(net.parameters())[0].grad.clone().flatten()

                #     # Calculate cosine similarity
                #     sim = F.cosine_similarity(grad_c.unsqueeze(0), grad_a.unsqueeze(0)).item()
                #     epoch_cos_sims.append(sim)

                # Actual training
                optimizer.zero_grad()
                
                outputs = net(inputs_imputed)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # FedMSplit regularization (eq.9, only if neighbour model is provided)
            if cfg.neighbour_model and cfg.lambda_reg > 0:
                proximal_term = 0.0
                for name, param in net.named_parameters():
                    if name in neighbour_params:
                        target = neighbour_params[name]
                        proximal_term += torch.sum((param - target) ** 2)
                loss += (cfg.lambda_reg / 2.0) * proximal_term

            # Backpropagation
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            n_batches += 1

        if not cfg.mt_enabled and len(epoch_cos_sims) > 0:
            avg_sim = sum(epoch_cos_sims) / len(epoch_cos_sims)
            print(f"Epoch Avg Gradient Cosine Similarity: {avg_sim:.4f}")

    avg_trainloss = running_loss / max(1, n_batches)
    train_acc = correct / total if total > 0 else 0.0

    return avg_trainloss, train_acc


def test(
        net: Module, testloader: DataLoader, cfg: TrainConfig
) -> Tuple[float, float]:
    """Validate the model on the test set."""
    net.to(cfg.device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss, n_batches = 0, 0, 0.0, 0

    with torch.no_grad():
        for inputs_imputed, labels, is_augmented in testloader:
            labels = labels.to(cfg.device)
            inputs_imputed = {k: v.to(cfg.device) for k, v in inputs_imputed.items()}
            aug_mask = is_augmented.to(cfg.device, dtype=torch.bool)
            clean_mask = ~aug_mask
            batch_size = labels.size(0)

            # 2 head architecture logic
            if cfg.mt_enabled:
                # Empty tensor to hold all predictions for the batch
                batch_logits = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)

                # Dynamic routing: clean -> primary, imputed -> aux
                if clean_mask.any():
                    clean_dict = {k: v[clean_mask] for k, v in inputs_imputed.items()}
                    batch_logits[clean_mask] = net(clean_dict, use_auxiliary=False)
                
                if aug_mask.any():
                    aug_dict = {k: v[aug_mask] for k, v in inputs_imputed.items()}
                    batch_logits[aug_mask] = net(aug_dict, use_auxiliary=True)
                
                outputs = batch_logits
            else:
                outputs = net(inputs_imputed)

            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            n_batches += 1

    accuracy = (correct / total) if total > 0 else 0.0
    avg_loss = loss / max(1, n_batches)
    return avg_loss, accuracy
