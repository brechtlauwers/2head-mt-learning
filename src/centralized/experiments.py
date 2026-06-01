import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from core.models import SplitNetwork, MultiTaskModel
from core.data import (
    SingleViewWrapper, DropOneViewWrapper
)

from centralized.utils import get_dataset_details, count_params
from centralized.data_setup import get_dataloaders
from centralized.train import train_and_eval, train_and_eval_mt

from centralized.config import DEVICE


def run_experiment(
        dataset_name: str, experiment_type: str, missing_rate: float,
        seed: int, use_validation: bool = True, fusion_type: str = "mid_late"
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Running {experiment_type} on {dataset_name}")

    num_classes, hidden_dim, batch_size, _, _, lmbda, lr = get_dataset_details(dataset_name)

    trainloader, valloader, testloader, sensor_dims = get_dataloaders(  # pyright: ignore
        dataset_name, experiment_type, missing_rate, seed,
        batch_size, use_validation=use_validation
    )

    assert isinstance(sensor_dims, dict)

    # TOTAL TRAINING EPOCHS
    epochs = 20

    if "2head" in experiment_type:
        model = MultiTaskModel(
            sensor_dims=sensor_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            fusion_type=fusion_type
        )
        print(f"Model size: {count_params(model):.3f}M Parameters")
        final_acc, grad_sim_history = train_and_eval_mt(
            model, trainloader, testloader, valloader=valloader,
            epochs=epochs, lr=lr, lmbda=lmbda
        )
    else:
        model = SplitNetwork(
            sensor_dims=sensor_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            fusion_type=fusion_type
        )
        print(f"Model size: {count_params(model):.3f}M Parameters")
        final_acc, grad_sim_history = train_and_eval(
            model, trainloader, testloader, valloader=valloader,
            epochs=epochs, lr=lr
        )

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), grad_sim_history, marker='o', color='purple', linewidth=2)
    plt.title(f"Gradient Sim over time\n({dataset_name}, {experiment_type}, Missing: {missing_rate*100}%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Cosine Similarity", fontsize=10)
    
    plt.ylim(0.0, 1.05) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    safe_type = experiment_type.replace("2head_", "")
    os.makedirs("./grad_sim", exist_ok=True)
    plt.savefig(f"./grad_sim/grad_sim_{dataset_name}_{safe_type}_rate{missing_rate}_seed{seed}.png", dpi=300)
    plt.close()

    return final_acc


def generate_results(dataset_name: str, experiment_type: str, seeds: list[int], fusion_type: str = "mid_late"):
    missing_rates = [0.0, 0.1, 0.3, 0.5, 0.7]

    results_base = {rate: [] for rate in missing_rates}
    results_mt = {rate: [] for rate in missing_rates}

    for rate in missing_rates:
        for seed in seeds:
            print(f"\nMissing rate: {rate*100}%, Seed: {seed}")

            if rate == 0.0:
                acc_base = run_experiment(
                    dataset_name, "baseline", rate, seed, fusion_type=fusion_type
                )
                results_base[rate].append(acc_base)
                print(f"-> No missing baseline acc: {acc_base*100:.2f}%")
                
                acc_mt = run_experiment(
                    dataset_name, "2head_baseline", rate, seed, fusion_type=fusion_type
                )
                results_mt[rate].append(acc_mt)
                print(f"-> No missing multitask acc: {acc_mt*100:.2f}%")
            else:
                # Run base model (1 head architecture)
                acc_base = run_experiment(
                    dataset_name, f"impute_{experiment_type}", rate, seed, fusion_type=fusion_type
                )
                results_base[rate].append(acc_base)
                print(f"-> Baseline peak acc: {acc_base*100:.2f}%")

                # Run multi task model (2 head architecture)
                acc_mt = run_experiment(
                    dataset_name, f"2head_impute_{experiment_type}", rate, seed, fusion_type=fusion_type
                )
                results_mt[rate].append(acc_mt)
                print(f"-> Multitask peak acc: {acc_mt*100:.2f}%")

    print("FINAL TABLE")
    print("| Missing Rate | 1-Head Baseline (mean +/- std) | 2-Head Multi-Task (mean +/- std) | Delta |")

    means_base, means_mt = [], []
    stds_base, stds_mt = [], []

    for rate in missing_rates:
        # Calculate stats
        mean_b = np.mean(results_base[rate]) * 100
        std_b = np.std(results_base[rate]) * 100
        means_base.append(mean_b)
        stds_base.append(std_b)

        mean_m = np.mean(results_mt[rate]) * 100
        std_m = np.std(results_mt[rate]) * 100
        means_mt.append(mean_m)
        stds_mt.append(std_m)

        delta = mean_m - mean_b

        print(f"| {int(rate*100)}% | {mean_b:.2f}% +/- {std_b:.2f}% | "
              f"{mean_m:.2f}% +/- {std_m:.2f}% | {'+' if delta > 0 else ''}{delta:.2f}% |")

    means_base = np.array(means_base)
    stds_base = np.array(stds_base)
    means_mt = np.array(means_mt)
    stds_mt = np.array(stds_mt)

    # Generate graph
    plt.figure(figsize=(8, 5))
    missing_rates_pct = np.array(missing_rates) * 100
    
    # Plot main solid lines
    plt.plot(missing_rates_pct, means_base, marker='o', linestyle='dashed', color='red', label=f'1-Head Baseline {experiment_type}')
    plt.plot(missing_rates_pct, means_mt, marker='s', linestyle='solid', color='blue', label=f'2-Head Multi-Task {experiment_type}')

    # Add transparent variance regions
    plt.fill_between(missing_rates_pct, means_base - stds_base, means_base + stds_base, color='red', alpha=0.15)
    plt.fill_between(missing_rates_pct, means_mt - stds_mt, means_mt + stds_mt, color='blue', alpha=0.15)

    # Fix x ticks
    plt.xticks(missing_rates_pct, [f"{int(r)}%" for r in missing_rates_pct])

    plt.title(f"Accuracy vs. Missing modality rate ({dataset_name.capitalize()})", fontsize=14)
    plt.xlabel("Missing rate", fontsize=12)
    plt.ylabel("Test accuracy (%) at best validation", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"results_{dataset_name}_{experiment_type}.png", dpi=300)
    plt.close()


def study_mod_dominance(dataset_name: str, seeds: list[int], epochs: int = 20):
    """
    Train and eval model on each modality individually to get predictive
    power of each modality.
    """
    print(f"\nStarting modality dominance study for: {dataset_name.upper()}")

    num_classes, hidden_dim, batch_size, _, _, _, lr = get_dataset_details(dataset_name)

    # Just get sensor_dims with one seed
    _, _, _, sensor_dims = get_dataloaders(
        dataset_name, "baseline", missing_rate=0.0, seed=seeds[0],
        batch_size=batch_size, use_validation=True
    )
    assert isinstance(sensor_dims, dict)

    # Store list of accuracies for each view
    results_history = {v: [] for v in sensor_dims.keys()}

    # Outer loop. Iterate over all seeds
    for seed in seeds:
        print(f"\nRunning seed: {seed}")

        # For determinism
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Get complete dataloaders for this seed
        trainloader_clean, valloader_clean, testloader_clean, _ = get_dataloaders(
            dataset_name, "baseline", missing_rate=0.0, seed=seed,
            batch_size=batch_size, use_validation=True
        )

        # Inner loop. Iterate through all single views
        for target_view in sensor_dims.keys():
            print(f"\nTraining exclusively on view {target_view}")

            # Wrap the clean datasets to isolate the target view
            train_ds = SingleViewWrapper(trainloader_clean.dataset, target_view)
            test_ds = SingleViewWrapper(testloader_clean.dataset, target_view)
            if valloader_clean is not None:
                val_ds = SingleViewWrapper(valloader_clean.dataset, target_view)
            else:
                val_ds = None

            # Create dataloaders for the isolated data
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=(DEVICE.type == "cuda"))
            if val_ds:
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(DEVICE.type == "cuda"))
            else:
                val_loader = None

            # Initialize baseline model for this specific view
            model = SplitNetwork(
                sensor_dims=sensor_dims,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )

            # Train and eval
            final_acc, _ = train_and_eval(
                model, train_loader, test_loader, valloader=val_loader,
                epochs=epochs, lr=lr
            )

            results_history[target_view].append(final_acc)

    # Print results
    final_results = {}
    print("MODALITY DOMINANCE RESULTS:")

    for view, acc_list in results_history.items():
        mean_acc = np.mean(acc_list) * 100
        std_acc = np.std(acc_list) * 100
        final_results[view] = {"mean": mean_acc, "std": std_acc, "raw": acc_list}
        print(f"View {view} isolated accuracy:\t{mean_acc:.2f}% +/- {std_acc:.2f}%")
        
    # Identify the strongest modality based on the mean acc
    dominant_view = max(final_results, key=lambda k: final_results[k]["mean"])
    print(f"\nConclusion: View {dominant_view} is the dominant modality for this task.")

    return final_results


def inference_view_drop(dataset_name: str, seeds: list[int], epochs: int = 20):
    """
    Train model on complete multi-view data, but remove (zero-out) individual
    views during inference to test fusion robustness.
    """
    print(f"\nStarting inference view dropping test for: {dataset_name.upper()}")

    num_classes, hidden_dim, batch_size, _, _, _, lr = get_dataset_details(dataset_name)

    # Just get sensor_dims with one seed
    _, _, _, sensor_dims = get_dataloaders(
        dataset_name, "baseline", missing_rate=0.0, seed=seeds[0],
        batch_size=batch_size, use_validation=True
    )
    assert isinstance(sensor_dims, dict)

    # Store list of accuracies for each view, and one for the clean baseline
    results_history = {v: [] for v in sensor_dims.keys()}
    clean_history = []

    # Outer loop. Iterate over all seeds
    for seed in seeds:
        print(f"\nRunning seed: {seed}")

        # For determinism
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Get complete dataloaders for this seed
        trainloader_clean, valloader_clean, testloader_clean, _ = get_dataloaders(
            dataset_name, "baseline", missing_rate=0.0, seed=seed,
            batch_size=batch_size, use_validation=True
        )

        # Initialize baseline model for all views
        model = SplitNetwork(
            sensor_dims=sensor_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        # Train and eval on clean data
        clean_test_acc, _ = train_and_eval(
            model, trainloader_clean, testloader_clean, valloader=valloader_clean,
            epochs=epochs, lr=lr
        )
        clean_history.append(clean_test_acc)

        # Helper function for inference only
        def test_only(ablated_loader):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs_dict, labels, _ in ablated_loader:
                    labels = labels.to(DEVICE)
                    inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
                    logits = model(inputs_dict)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / max(1, total)

        # Inner loop. Iterate through all single views to test them during inference
        print("Starting inference testing...")
        for target_view in sensor_dims.keys():
            print(f"Testing inference with view {target_view} completely missing")

            # Wrap the clean test dataset to drop the target view
            test_ds_ablated = DropOneViewWrapper(testloader_clean.dataset, target_view)
            
            # Create dataloader for the tested data
            ablated_test_loader = DataLoader(
                test_ds_ablated, batch_size=batch_size, 
                shuffle=False, pin_memory=(DEVICE.type == "cuda")
            )

            # Evaluate the frozen model on the broken data
            ablated_acc = test_only(ablated_test_loader)
            results_history[target_view].append(ablated_acc)

    # Print results
    final_results = {}
    print("\nINFERENCE VIEW DROPPING RESULTS:")
    
    mean_clean = np.mean(clean_history) * 100
    std_clean = np.std(clean_history) * 100
    print(f"Original all-view acc:\t{mean_clean:.2f}% +/- {std_clean:.2f}%\n")

    for view, acc_list in results_history.items():
        mean_acc = np.mean(acc_list) * 100
        std_acc = np.std(acc_list) * 100
        final_results[view] = {"mean": mean_acc, "std": std_acc, "raw": acc_list}
        
        drop = mean_clean - mean_acc
        print(f"If View {view} fails accuracy:\t{mean_acc:.2f}% +/- {std_acc:.2f}% (drop of {drop:.2f}%)")

    return final_results


def drop_missing_experiment(dataset_name: str, seeds: list[int], epochs: int = 20):
    """
    Train on only complete samples (dropping incomplete ones),
    and tests on zero-imputed test set to simulate a failure.
    Would not make sense to use 2-head for this one.
    """
    missing_rates = [0.0, 0.1, 0.3, 0.5, 0.7]

    print(f"Starting drop missing experiment for: {dataset_name.upper()}")

    results = {rate: [] for rate in missing_rates}

    for rate in missing_rates:
        for seed in seeds:
            print(f"\nMissing rate: {rate*100}%, Seed: {seed}")

            # For determinism
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            if rate == 0.0:
                acc = run_experiment(dataset_name, "baseline", rate, seed)
                results[rate].append(acc)
            else:
                num_classes, hidden_dim, batch_size, _, _, _, lr = get_dataset_details(dataset_name)

                # Get the dropped training data
                trainloader_dropped, valloader_dropped, _, sensor_dims = get_dataloaders(
                    dataset_name, "drop_missing", rate, seed, batch_size, use_validation=True
                )
                assert isinstance(sensor_dims, dict)

                # Get the zero-imputed test data
                _, _, testloader_imputed, _ = get_dataloaders(
                    dataset_name, "impute_zero", rate, seed, batch_size, use_validation=False
                )

                # Train 1-head model
                model = SplitNetwork(
                    sensor_dims=sensor_dims,
                    hidden_dim=hidden_dim,
                    num_classes=num_classes
                )
                acc, _ = train_and_eval(
                    model, trainloader_dropped, testloader_imputed, valloader=valloader_dropped,
                    epochs=epochs, lr=lr
                )
                results[rate].append(acc)

    # Print Final Table
    print(f"\nDROP MISSING RESULTS TABLE ({dataset_name.upper()})")
    print("| Missing Rate | 1-Head Drop Missing |")

    final_results = {}
    for rate in missing_rates:
        mean_b = np.mean(results[rate]) * 100
        std_b = np.std(results[rate]) * 100

        final_results[rate] = {"mean": mean_b, "std": std_b}

        print(f"| {int(rate*100)}% | {mean_b:.2f}% +/- {std_b:.2f}% |")

    return final_results


def plot_gradient_similarity(dataset_name: str, missing_rate: float, seed: int, epochs: int = 20):
    """
    Gradient similarity vs. imputation quality.
    Trains the 2-Head model under four different imputation strategies at a fixed 
    missing rate and plots their gradient cosine similarity over time.
    """
    print(f"Dataset: {dataset_name.upper()} | Missing Rate: {missing_rate*100}% | Seed: {seed}")

    num_classes, hidden_dim, batch_size, _, _, lmbda, lr = get_dataset_details(dataset_name)

    # The four imputation methods to compare
    imputation_methods = {
        "impute_zero": {"label": "Zero Imputation", "color": "red", "linestyle": "dashed"},
        "impute_gaus": {"label": "Gaussian Imputation", "color": "orange", "linestyle": "dashdot"},
        "impute_dcp": {"label": "DCP Imputation", "color": "green", "linestyle": "dotted"},
        "impute_cpm": {"label": "CPM-Nets Imputation", "color": "blue", "linestyle": "solid"}
    }

    all_histories = {}

    for exp_type, style in imputation_methods.items():
        print(f"\nEvaluating: {style['label']}")
        
        # Get the dataloaders
        trainloader, valloader, testloader, sensor_dims = get_dataloaders(
            dataset_name, exp_type, missing_rate, seed, batch_size, use_validation=True
        )
        assert isinstance(sensor_dims, dict)
        
        # Initialize a fresh model
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SplitNetwork(
            sensor_dims=sensor_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            fusion_type="mid_late"
        )

        # Train and capture the gradient history
        _, grad_sim_history = train_and_eval(
            model, trainloader, testloader, valloader=valloader,
            epochs=epochs, lr=lr
        )

        all_histories[exp_type] = grad_sim_history

    # Plot
    print("\nGenerating Gradient Similarity Plot...")
    plt.figure(figsize=(8, 5))
    
    x_axis = range(1, epochs + 1)
    for exp_type, history in all_histories.items():
        style = imputation_methods[exp_type]
        plt.plot(
            x_axis, history, 
            label=style['label'], 
            color=style['color'], 
            linestyle=style['linestyle'], 
            linewidth=2,
            marker='o', markersize=4
        )

    plt.title(f"Gradient alignment by imputation quality\n({dataset_name.capitalize()}, {int(missing_rate*100)}% Missing)", fontsize=13)
    plt.xlabel("Training epoch", fontsize=11)
    plt.ylabel("Cosine similarity (clean vs. imputed)", fontsize=11)
    plt.xticks(range(0, epochs + 1, 5))

    plt.ylim(0.0, 1.05) 
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs("./grad_sim", exist_ok=True)
    filename = f"./grad_sim/exp1_imputation_quality_{dataset_name}_rate{missing_rate}_seed{seed}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved to: {filename}")


def plot_gradsim_vs_missing(dataset_name: str, imputation_type: str, seed: int, epochs: int = 20):
    """
    Gradient similarity vs. missing rate
    Trains the 1-Head baseline under a specific imputation strategy across
    increasing missing rates to show how volume exacerbates gradient conflict.

    This function will only work correctly if the gradient calculation is correct!
    """
    print(f"Dataset: {dataset_name.upper()} | Imputation: {imputation_type} | Seed: {seed}")

    num_classes, hidden_dim, batch_size, _, _, _, lr = get_dataset_details(dataset_name)

    missing_rates = [0.1, 0.3, 0.5, 0.7]
    rate_styles = {
        0.1: {"label": "10% Missing", "color": "#ff9999", "linestyle": "solid"},
        0.3: {"label": "30% Missing", "color": "#ff4d4d", "linestyle": "dashed"},
        0.5: {"label": "50% Missing", "color": "#cc0000", "linestyle": "dashdot"},
        0.7: {"label": "70% Missing", "color": "#660000", "linestyle": "dotted"}
    }

    all_histories = {}

    for rate in missing_rates:
        print(f"\nEvaluating: {int(rate*100)}% Missing Rate")
        
        trainloader, valloader, testloader, sensor_dims = get_dataloaders(
            dataset_name, f"impute_{imputation_type}", rate, seed, batch_size, use_validation=True
        )
        assert isinstance(sensor_dims, dict)
        
        # Initialize the 1-head model to measure conflict
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SplitNetwork(
            sensor_dims=sensor_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            fusion_type="mid_late"
        )

        # Train and capture the gradient history
        _, grad_sim_history = train_and_eval(
            model, trainloader, testloader, valloader=valloader,
            epochs=epochs, lr=lr
        )
        
        all_histories[rate] = grad_sim_history

    # Plot
    print("\nGenerating Gradient Similarity Plot...")
    plt.figure(figsize=(8, 5))
    
    x_axis = range(1, epochs + 1)
    for rate, history in all_histories.items():
        style = rate_styles[rate]
        plt.plot(
            x_axis, history, 
            label=style['label'], 
            color=style['color'], 
            linestyle=style['linestyle'], 
            linewidth=2.5,
            marker='o', markersize=4
        )

    plt.title(f"Gradient conflict scaling by missing rate\n({dataset_name.capitalize()}, {imputation_type.capitalize()})", fontsize=13)
    plt.xlabel("Training epoch", fontsize=11)
    plt.ylabel("Cosine similarity (clean vs. total)", fontsize=11)
    plt.xticks(range(0, epochs + 1, 5))
    
    # Allow y-axis to drop below zero if the conflict is highly destructive
    plt.ylim(-0.2, 1.05) 
    
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs("./grad_sim", exist_ok=True)
    filename = f"./grad_sim/exp2_missing_scaling_{dataset_name}_{imputation_type}_seed{seed}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Plot saved to: {filename}")


def plot_lambda_sensitivity(dataset_name: str, imputation_type: str = "impute_zero", seed: int = 50, epochs: int = 20):
    """
    Runs a targeted lambda sensitivity analysis to prove the 'data starvation' hypothesis
    """
    missing_rates = [0.1, 0.3, 0.5, 0.7]
    lambda_values = [0.1, 0.3, 0.7, 1.0, 2.0, 4.0]

    # Convert to labels to space them evenly on axis
    x_labels = [str(lam) for lam in lambda_values]

    rate_styles = {
        0.1: {"label": "10% Missing", "color": "#ff9999", "linestyle": "solid"},
        0.3: {"label": "30% Missing", "color": "#ff4d4d", "linestyle": "dashed"},
        0.5: {"label": "50% Missing", "color": "#cc0000", "linestyle": "dashdot"},
        0.7: {"label": "70% Missing", "color": "#660000", "linestyle": "dotted"}
    }
    
    # Get base hyperparams (override lambda afterwards)
    num_classes, hidden_dim, batch_size, _, _, _, lr = get_dataset_details(dataset_name)
    
    results = {rate: [] for rate in missing_rates}
    
    for rate in missing_rates:
        print(f"\n--- Evaluating {int(rate*100)}% Missing Rate ---")
         
        # Load data once per missing rate to save time
        trainloader, valloader, testloader, sensor_dims = get_dataloaders(
            dataset_name, imputation_type, rate, seed, batch_size, use_validation=True
        )
        assert isinstance(sensor_dims, dict)
        
        for lmbda in lambda_values:
            print(f"Training with lambda = {lmbda}...")
            
            # Reset seeds for a fair comparison
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            model = MultiTaskModel(
                sensor_dims=sensor_dims,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                fusion_type="mid_late"
            )
            
            # Run the training with the specific sweep lambda
            final_acc, _ = train_and_eval_mt(
                model, trainloader, testloader, valloader=valloader,
                epochs=epochs, lr=lr, lmbda=lmbda
            )
            
            results[rate].append(final_acc * 100) # Store as percentage

    # Plot
    print("\nGenerating Lambda Sensitivity Plot...")
    plt.figure(figsize=(7, 5))

    for rate, history in results.items():
        style = rate_styles[rate]
        plt.plot(
            x_labels, history, 
            label=style['label'], 
            color=style['color'], 
            linestyle=style['linestyle'], 
            linewidth=2.5,
            marker='o', markersize=5
        )
    
    plt.title(r"Impact of auxiliary loss weight ($\lambda$) on accuracy" + f"\n({dataset_name.capitalize()}, {imputation_type.replace('impute_', '').capitalize()} imputation)", fontsize=13)
    plt.xlabel(r"Auxiliary loss weight ($\lambda$)", fontsize=11)
    plt.ylabel("Test accuracy (%)", fontsize=11)
    
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs("./grad_sim", exist_ok=True)
    filename = f"./grad_sim/lambda_sensitivity_{dataset_name}_{imputation_type}_seed{seed}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Plot saved to: {filename}")
