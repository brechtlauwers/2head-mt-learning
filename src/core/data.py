from typing import Tuple, List, Dict, Any
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from flwr.app import Context
from core.datasets import (
    ModelNet40Features,
    VehicleSensITFeatures
)
from core.utils import load_experiment_config
from core.cpm_nets import CPMNet_Works

"""
Handles everything related to data before the model sees it. It retrieves the raw
data, splits it over the clients, generates missing matrices, and applies the
dataset wrappers (zero, noise, CPM)
"""

# Sizes of train data split
DATASET_SIZE_MAP = {
    "modelnet40": 9843,
    "sensit": 39444,
}

# Cache FederatedDataset
fds = None

class NoMissingWrapper(Dataset):
    """
    Wrapper for complete datasets.
    Just adds the 'is_augmented' label (always False)
    """
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        data_dict, label = self.base_dataset[idx]
        is_augmented = False
        return data_dict, label, is_augmented


class ZeroImputeWrapper(Dataset):
    """Fill-in the missing views with zero's"""
    def __init__(self, base_dataset, missing_matrix):
        self.base_dataset = base_dataset
        self.missing_matrix = missing_matrix

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        data_dict, label = self.base_dataset[idx]
        imputed_dict = {}

        # Check if sample has any missing values
        is_augmented = np.any(self.missing_matrix[:, idx])

        for v in range(self.missing_matrix.shape[0]):
            key = f"view_{v}"
            # If the matrix says this view is missing for this sample, just return zeros
            if self.missing_matrix[v, idx]:
                imputed_dict[key] = torch.zeros_like(data_dict[key])
            else:
                imputed_dict[key] = data_dict[key]

        return imputed_dict, label, is_augmented


class GausImputeWrapper(Dataset):
    """
    Fill missing views with L2-normalized Gaussian noise.
    It matches the distribution scale of the original data.
    """
    def __init__(self, base_dataset, missing_matrix, intensity=1.0):
        self.base_dataset = base_dataset
        self.missing_matrix = missing_matrix
        self.intensity = intensity

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        rng = torch.Generator()
        rng.manual_seed(idx + 99999)
        
        data_dict, label = self.base_dataset[idx]
        imputed_dict = {}

        # Check if sample has any missing values
        is_augmented = np.any(self.missing_matrix[:, idx])

        for v in range(self.missing_matrix.shape[0]):
            key = f"view_{v}"

            if self.missing_matrix[v, idx]:
                # Generate random Gaussian noise
                dim = data_dict[key].shape[0]
                noise = torch.randn(dim, generator=rng)

                # Normalize it
                noise = torch.nn.functional.normalize(noise, p=2, dim=0)

                # Apply intensity scaling
                imputed_dict[key] = noise * self.intensity
            else:
                imputed_dict[key] = data_dict[key]

        return imputed_dict, label, is_augmented

class SingleViewWrapper(Dataset):
    """Simulate a single view scenario by keeping only 1 view"""
    def __init__(self, base_dataset: Dataset, target_view: int) -> None:
        self.base_dataset = base_dataset
        self.target_view = target_view

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        inputs_dict = item[0]
        label = item[1]
        extra = item[2] if len(item) > 2 else False

        isolated_dict = {}
        for k, v in inputs_dict.items():
            key_str = str(k).replace("view_", "")
            
            if key_str == str(self.target_view):
                # Keep this view
                isolated_dict[k] = v
            else:
                # Delete other views with zeros
                isolated_dict[k] = torch.zeros_like(v)

        return isolated_dict, label, extra

class DropOneViewWrapper(Dataset):
    """Simulate a sensor failure at inference time by destroying one view with zeros"""
    def __init__(self, base_dataset: Dataset, view_to_drop: int) -> None:
        self.base_dataset = base_dataset
        self.view_to_drop = view_to_drop

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        inputs_dict = item[0]
        label = item[1]
        extra = item[2] if len(item) > 2 else False

        ablated_dict = {}
        for k, v in inputs_dict.items():
            key_str = str(k).replace("view_", "")
            
            if key_str == str(self.view_to_drop):
                # Drop this view, replace by zeros
                ablated_dict[k] = torch.zeros_like(v)
            else:
                # Keep all other views perfectly intact
                ablated_dict[k] = v

        return ablated_dict, label, extra


class CPMIndexDataset(Dataset):
    def __init__(self, base_dataset, missing_matrix):
        self.base_dataset = base_dataset
        self.missing_matrix = missing_matrix
        self.num_views = missing_matrix.shape[0]
        
        # Precompute the sn dictionary (1: available, 0: missing)
        # sn -> Sample availability matrix, inverse of the missing matrix
        self.sn_matrix = (~self.missing_matrix).astype(np.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data_dict, label = self.base_dataset[idx]
        sn = {str(v): self.sn_matrix[v, idx] for v in range(self.num_views)}
        return idx, data_dict, label, sn


class CPMImputeWrapper(Dataset):
    """Fill missing views using trained CPM model"""
    def __init__(self, base_dataset, missing_matrix, cpm_model, mode="train"):
        self.base_dataset = base_dataset
        self.missing_matrix = missing_matrix
        self.num_views = missing_matrix.shape[0]

        # pre-compute is_augmented mask for whole dataset
        self.is_augmented_array = np.any(self.missing_matrix, axis=0)

        # Get correct static memory bank once
        h_bank = cpm_model.h_train if mode == "train" else cpm_model.h_test

        # Pre-decode the entire dataset at once
        print(f"CPMImputeWrapper: Pre-calculating all {len(base_dataset)} {mode} reconstructions...")
        with torch.no_grad():
            # Pass the entire (N, latent_dim) matrix through the decoder
            all_recons_gpu = cpm_model.calculate(h_bank)
            self.all_recons = {}
            for v in range(self.num_views):
                self.all_recons[str(v)] = all_recons_gpu[str(v)].cpu()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data_dict, label = self.base_dataset[idx]
        imputed_dict = {}

        # Check if sample has any missing values
        is_augmented = self.is_augmented_array[idx]

        for v in range(self.num_views):
            key = f"view_{v}"
            if self.missing_matrix[v, idx]:
                # Instant memory lookup
                imputed_dict[key] = self.all_recons[str(v)][idx]
            else:
                imputed_dict[key] = data_dict[key]

        return imputed_dict, label, is_augmented


class DCPImputeWrapper(Dataset):
    """Fill missing views with Dual Contrastive Prediction (DCP)"""
    def __init__(self, base_dataset, missing_matrix, dcp_model):
        self.base_dataset = base_dataset
        self.missing_matrix = missing_matrix
        self.num_views = missing_matrix.shape[0]
        device = next(dcp_model.parameters()).device

        self.dcp_model = dcp_model

        # pre-compute is_augmented mask for whole dataset
        self.is_augmented_array = np.any(self.missing_matrix, axis=0)

        # Containers for pre-computed dataset
        self.precomputed_data = {str(v): [] for v in range(self.num_views)}
        self.labels = []

        print(f"DCPImputeWrapper: Pre-calculating {len(base_dataset)} missing views.")

        with torch.no_grad():
            for idx in range(len(base_dataset)):
                data_dict, label = self.base_dataset[idx]
                self.labels.append(label)

                # Identify available and missing views for this sample
                available_views = [v for v in range(self.num_views) if not self.missing_matrix[v, idx]]
                missing_views = [v for v in range(self.num_views) if self.missing_matrix[v, idx]]

                # Encode available
                available_latents = {}
                for j in available_views:
                    key = f"view_{j}"
                    autoencoder_j = self.dcp_model.encoders[str(j)]
                    gpu_input = data_dict[key].unsqueeze(0).to(device)
                    available_latents[j] = autoencoder_j.encoder(gpu_input).cpu().detach()

                # Impute missing
                recons = {}
                for i in missing_views:
                    predictions = []
                    for j in available_views:
                        predictor = self.dcp_model.predictors[f"{j}_to_{i}"]
                        z_i_pred, _ = predictor(available_latents[j].to(device))
                        predictions.append(z_i_pred)
                    
                    fused_z = torch.stack(predictions).mean(dim=0)
                    autoencoder_i = self.dcp_model.encoders[str(i)]
                    # Squeeze the batch dimension so it matches the raw data shape
                    recons[str(i)] = autoencoder_i.decoder(fused_z).squeeze(0).cpu().detach()

                # Store the final result for this specific sample
                for v in range(self.num_views):
                    key = f"view_{v}"
                    if self.missing_matrix[v, idx]:
                        self.precomputed_data[str(v)].append(recons[str(v)])
                    else:
                        self.precomputed_data[str(v)].append(data_dict[key])

        # Convert lists to stacked tensors for faster lookup
        for v in range(self.num_views):
            self.precomputed_data[str(v)] = torch.stack(self.precomputed_data[str(v)])
        

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        imputed_dict = {}
        
        # O(1) lookup
        for v in range(self.num_views):
            imputed_dict[f"view_{v}"] = self.precomputed_data[str(v)][idx]
            
        return imputed_dict, self.labels[idx], self.is_augmented_array[idx]


def generate_missing_matrix(
        num_samples: int, num_modalities: int, active_sensors: List[int], 
        sample_missing_rate: float, seed: int
) -> np.ndarray:
    """
    Creates a unified missing matrix (V x N).
    True -> missing, False -> present
    Combines client-level and sample-level missingness.
    """
    rng = np.random.default_rng(seed)
    matrix = np.zeros((num_modalities, num_samples), dtype=bool)

    # Apply client-level missingness
    # If a sensor is not active for this client, all samples are missing that view
    for v in range(num_modalities):
        if v not in active_sensors:
            matrix[v, :] = True

    # Apply sample-level random missingness
    # Only drop views from sensors the client actually possesses
    if sample_missing_rate > 0.0:
        for v in active_sensors:
            matrix[v, :] = rng.random(num_samples) < sample_missing_rate

    # Guarantee at least 1 view per sample
    all_missing_mask = np.all(matrix, axis=0)
    num_all_missing = np.sum(all_missing_mask)

    if num_all_missing > 0:
        # Only recover a view using a sensor the client actually owns
        keep_indices = rng.choice(active_sensors, size=num_all_missing)
        missing_sample_indices = np.where(all_missing_mask)[0]
        matrix[keep_indices, missing_sample_indices] = False

    return matrix

def _calculate_partitions(
        global_len: int, num_clients: int, config: Dict[str, Any], seed: int = 2025
) -> Tuple[List[int], List[List[int]]]:
    """
    Helper to calculate client data size partitions using config dictionary.
    Client sizes are sampled from Gaussian distribution.
    Also returns mask of missing sensors per client.
    """
    # fixed random generator for determinism
    rng = np.random.default_rng(seed)
    mu = config['dataset']['sample_size_mean']
    sigma = config['dataset']['sample_size_std']
    num_modalities = config['dataset']['num_modalities']

    # Calculate client sizes (gaussian) until global sum fits
    client_sizes = rng.normal(loc=mu, scale=sigma, size=num_clients)
    # at least one available sensor
    client_sizes = np.maximum(client_sizes, 1.0)
    # Normalize to sum to 1.0
    proportions = client_sizes / np.sum(client_sizes)
    # Scale to global length
    client_sizes = (global_len * proportions).astype(int)

    # Calculate remainder (remainder due to rounding)
    current_sum = np.sum(client_sizes)
    diff = int(global_len - current_sum)
    for i in range(abs(diff)):
        if diff > 0:
            client_sizes[i % num_clients] += 1
        else:
            client_sizes[i % num_clients] = max(
                1, client_sizes[i % num_clients] - 1
            )

    # Client level sensor missingness
    client_missing_rate = float(config["dataset"]["client_sensor_missing_rate"])
    sensor_prob = 1.0 - client_missing_rate

    active_sensors = []
    for _ in range(num_clients):
        # Randomly set true/false for every sensors
        active_mask = rng.binomial(n=1, p=sensor_prob, size=num_modalities).astype(bool)

        # Guarantee at least 1 working sensors per client
        if not any(active_mask):
            active_mask[rng.integers(0, num_modalities)] = True
        active_sensors.append(np.where(active_mask)[0].tolist())

    return client_sizes, active_sensors

def get_partition_metadata(
        num_partitions: int, config: Dict[str, Any]
) -> Dict[str, List[int]]: 
    """Public accessor for server app to get sensor map without loading data"""
    dataset_name = str(config["dataset"]["name"])
    seed = int(config["strategy"]["seed"])
    
    if dataset_name == "modelnet40":
        total_len = DATASET_SIZE_MAP["modelnet40"]
    elif dataset_name == "sensit":
        total_len = DATASET_SIZE_MAP["sensit"]
    else:
        raise ValueError("Dataset cannot be found!")

    # Calculate partitions to get the exact sensor map
    _, active_sensors_list = _calculate_partitions(
        global_len=total_len,
        num_clients=num_partitions,
        config=config,
        seed=seed
    )

    return {str(i): sensors for i, sensors in enumerate(active_sensors_list)}

def load_data(
        partition_id: int, num_partitions: int, context: Context
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Load partitioned data for a client.
    partition_id refers to the id of a specific client.
    """
    # Only initialize federated dataset once
    global fds
    cfg = load_experiment_config(
        str(context.run_config["experiment_config"])
    )
    
    dataset_name = str(cfg["dataset"]["name"])
    dataset_root = str(cfg["dataset"]["root"])
    seed = int(cfg["strategy"]["seed"])
    rng = np.random.default_rng(seed)
    
    # Determinism
    torch.manual_seed(seed + partition_id)

    if fds is None:
        # Load raw features
        if dataset_name == "sensit":
            global_train = VehicleSensITFeatures(root_dir=dataset_root, split='train')
            global_test = VehicleSensITFeatures(root_dir=dataset_root, split='test')
        elif dataset_name == "modelnet40":
            global_train = ModelNet40Features(features_path=dataset_root, split='train')
            global_test = ModelNet40Features(features_path=dataset_root, split='test')
        else:
            raise ValueError("Dataset not found.")

        # Partition the data
        train_sizes, active_sensors_list = _calculate_partitions(
            len(global_train), num_partitions, cfg, seed=seed
        )

        # Calculate the ratio of train data for each client
        ratios = train_sizes / np.sum(train_sizes)
        test_sizes = (ratios * len(global_test)).astype(int)

        # At least 1 test sample
        test_sizes = np.maximum(test_sizes, 1)

        # Divide the remaining samples over the clients
        diff = len(global_test) - np.sum(test_sizes)
        for i in range(diff):
            if diff > 0:
                test_sizes[i % len(test_sizes)] += 1
            else:
                idx = i % len(test_sizes)
                test_sizes[idx] = max(1, test_sizes[idx] - 1)

        # Cache the RAW global datasets and the sensor lists
        fds = {
            "global_train": global_train,
            "global_test": global_test,
            "train_sizes": train_sizes,
            "test_sizes": test_sizes,
            "train_idx": rng.permutation(len(global_train)),
            "test_idx": rng.permutation(len(global_test)),
            "batch_size": cfg['training']['batch_size'],
            "active_sensors_list": active_sensors_list,
        }

    # Client specific extraction and wrapping
    # Give certain partition to the client, calculate offsets
    start_tr = np.sum(fds["train_sizes"][:partition_id])
    end_tr = start_tr + fds["train_sizes"][partition_id]
    start_te = np.sum(fds["test_sizes"][:partition_id])
    end_te = start_te + fds["test_sizes"][partition_id]

    client_raw_train = Subset(fds["global_train"], fds["train_idx"][start_tr:end_tr])
    client_raw_test = Subset(fds["global_test"], fds["test_idx"][start_te:end_te])

    # Get this specific client's sensors
    client_sensors = fds["active_sensors_list"][partition_id]

    # Generate missing matrices specific to this client's subset size
    sample_missing_rate = float(cfg["dataset"]["sample_view_missing_rate"])
    num_modalities = int(cfg["dataset"]["num_modalities"])
    client_seed = seed + partition_id

    train_matrix = generate_missing_matrix(
        len(client_raw_train), num_modalities, client_sensors, sample_missing_rate, client_seed
    )
    test_matrix = generate_missing_matrix(
        len(client_raw_test), num_modalities, client_sensors, sample_missing_rate, client_seed + 1000
    )

    # Wrap the subset with the generated matrix
    aug_type = str(cfg["augmentation"]["type"])
    intensity = float(cfg["augmentation"]["intensity"])

    if aug_type == "noise":
        client_train_ds = GausImputeWrapper(client_raw_train, train_matrix, intensity)
        client_test_ds = GausImputeWrapper(client_raw_test, test_matrix, intensity)
    elif aug_type == "zeros":
        client_train_ds = ZeroImputeWrapper(client_raw_train, train_matrix)
        client_test_ds = ZeroImputeWrapper(client_raw_test, test_matrix)
    elif aug_type == "cpm":
        print(f"[{dataset_name} | Client {partition_id}] Training local CPM-Net on {len(client_raw_train)} samples...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        sensor_dims = dict(enumerate(cfg["model"]["sensor_dims"]))
        hidden_dim = int(cfg["model"]["hidden_dim"])
        num_classes = int(cfg["model"]["num_classes"])

        cpm_model = CPMNet_Works(
            device=device,
            view_dims=sensor_dims,
            trainLen=len(client_raw_train),
            testLen=len(client_raw_test),
            lsd_dim=hidden_dim
        ).to(device)

        weights_file = f"cpm_weights_{dataset_name}_client_{partition_id}.pth"

        if os.path.exists(weights_file):
            print(f"[{dataset_name} | Client {partition_id}] Loading cached local CPM-Net...")
            cpm_model.load_state_dict(torch.load(weights_file, weights_only=True, map_location=device))
        else:
            print(f"[{dataset_name} | Client {partition_id}] Training local CPM-Net on {len(client_raw_train)} samples...")
            
            train_loader_cpm = DataLoader(
                CPMIndexDataset(client_raw_train, train_matrix), 
                batch_size=fds["batch_size"], shuffle=True
            )
            test_loader_cpm = DataLoader(
                CPMIndexDataset(client_raw_test, test_matrix), 
                batch_size=fds["batch_size"], shuffle=False
            )

            cpm_epochs = int(cfg["augmentation"].get("cpm_epochs", 15))
            cpm_model.train_model(train_loader_cpm, num_classes=num_classes, epoch=cpm_epochs)
            cpm_model.test_model(test_loader_cpm, epoch=cpm_epochs)
            
            # Save it to disk
            torch.save(cpm_model.state_dict(), weights_file)

        cpm_model.eval()

        total_missing = np.sum(train_matrix)
        print(f"[{dataset_name} | Client {partition_id}] Synthesizing {total_missing} missing views based on local latent space.")
        
        client_train_ds = CPMImputeWrapper(client_raw_train, train_matrix, cpm_model, mode="train")
        client_test_ds = CPMImputeWrapper(client_raw_test, test_matrix, cpm_model, mode="test")
    else:
        raise ValueError(f"Augmentation type {aug_type} is unknown.")
        
    # Load into dataLoader
    trainloader = DataLoader(client_train_ds, batch_size=fds["batch_size"], shuffle=True)
    testloader = DataLoader(client_test_ds, batch_size=fds["batch_size"], shuffle=False)

    # Check for iid/non-iid label distribution
    # class_counts = torch.zeros(int(cfg["model"]["num_classes"]))
    # for _, labels, _ in trainloader:
    #     counts = torch.bincount(labels, minlength=len(class_counts))
    #     class_counts += counts
    # print(f"[Client {partition_id}] Class distribution: {class_counts.int().tolist()}")

    return trainloader, testloader, client_sensors
