import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from centralized.data_setup import get_dataloaders
from core.data import generate_missing_matrix


def visualize_reconstruction(dataset_name: str, experiment_type: str, missing_rate: float = 0.5, seed: int = 50):
    """
    Find test sample where view was dropped for CPM or DCP.
    Compare imputed version vs real version on a plot.
    """
    print(f"\nStarting {experiment_type.upper()} visual check for: {dataset_name.upper()}")

    # Get the clean data
    _, _, testloader_clean, sensor_dims = get_dataloaders(
        dataset_name, "baseline", missing_rate=0.0, seed=seed, 
        batch_size=32, use_validation=True
    )

    # Get the imputed data
    _, _, testloader_imp, _ = get_dataloaders(
        dataset_name, f"impute_{experiment_type}", missing_rate=missing_rate,
        seed=seed, batch_size=32, use_validation=True
    )

    clean_ds = testloader_clean.dataset
    imp_ds = testloader_imp.dataset  

    # Bypass dataset and regenerate the exact test matrix
    test_len = len(clean_ds)  # type: ignore
    num_sensors = len(sensor_dims)
    active_sensors = list(range(num_sensors))

    missing_matrix = generate_missing_matrix(
        test_len, num_sensors, active_sensors, missing_rate, seed=seed+1
    )

    # Find sample where view 0 is missing for comparison
    view_to_check = 0
    missing_indices = np.where(missing_matrix[view_to_check])[0]

    if len(missing_indices) == 0:
        print(f"No missing samples found for view {view_to_check}!")
        return

    # Get first sample that has a missing view
    sample_idx = missing_indices[0] 

    # dataset returns: data_dict, label, is_augmented
    clean_item = clean_ds[sample_idx]
    real_dict = clean_item[0]
    true_label = clean_item[1]

    imp_item = imp_ds[sample_idx]
    imp_dict = imp_item[0]

    real_feature = real_dict[f"view_{view_to_check}"].cpu().numpy()
    gen_feature = imp_dict[f"view_{view_to_check}"].cpu().numpy()

    # Calculate numerical similarity with cosine sim
    mse = np.mean((real_feature - gen_feature)**2)
    cos_sim = F.cosine_similarity(
        torch.tensor(real_feature).unsqueeze(0), 
        torch.tensor(gen_feature).unsqueeze(0)
    ).item()

    print(f"Sample {sample_idx} (class {true_label}) | view {view_to_check}")
    print(f"-> Mean Squared Error: {mse:.4f}")
    print(f"-> Cosine Similarity:  {cos_sim:.4f}")

    # Plotting
    color = "red" if experiment_type == "cpm" else "green"

    plt.figure(figsize=(10, 4))
    plt.plot(real_feature, label="Ground truth (real)", color="blue", alpha=0.8, linewidth=1.5)
    plt.plot(gen_feature, label=f"{experiment_type.upper()} Generated (imputed)", color=color, alpha=0.8, linestyle="dashed", linewidth=1.5)

    plt.title(f"{experiment_type.upper()} Reconstruction check | {dataset_name.upper()} | View {view_to_check+1}\n"
              f"MSE: {mse:.4f} | Cosine Sim: {cos_sim:.4f}", fontsize=12)
    plt.xlabel("Feature dimension index", fontsize=10)
    plt.ylabel("Normalized value", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    file_name = f"{experiment_type}_check_{dataset_name}.png"
    plt.savefig(file_name, dpi=300)
    print(f"Saved visualization to {file_name}\n")


if __name__ == "__main__":
    visualize_reconstruction(dataset_name="har", experiment_type="cpm", missing_rate=0.5, seed=50)
    visualize_reconstruction(dataset_name="har", experiment_type="dcp", missing_rate=0.5, seed=50)
