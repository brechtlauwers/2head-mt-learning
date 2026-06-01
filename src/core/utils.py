from typing import Union, Dict, Any, List, Optional
import os
import numpy as np
import random
import tomllib
import torch
import json
from pathlib import Path
from flwr.app import MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from torch.types import Tensor
from torch.utils.data import Dataset

"""
Utility functions such as loading config files, augmenting features,
saving model results, ...
"""

def _recursive_update(
        base_dict: Dict[str, Any], update_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merges update_dict into base_dict."""
    for k, v in update_dict.items():
        if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
            _recursive_update(base_dict[k], v)
        else:
            base_dict[k] = v
    return base_dict

def load_experiment_config(
        config_path: Union[str, Path]
) -> Dict[str, Any]:
    """Load toml config with extra inherits support"""
    # Load the target file
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        
    # Check for inheritance inside config file
    if "inherits" in config:
        parent_paths = config.pop("inherits")

        if isinstance(parent_paths, str):
            parent_paths = [parent_paths]

        base_config: Dict[str, Any] = {}
        base_dir = os.path.dirname(config_path)

        # Load parents in order
        for parent in parent_paths:
            full_path = os.path.normpath(os.path.join(base_dir, parent))
            parent_config = load_experiment_config(full_path)
            base_config = _recursive_update(base_config, parent_config)

        # Merge current config on top of parents
        config = _recursive_update(base_config, config)
    return config

def zero_like_block(
        block_state_dict: Dict[str, Any]
) -> Dict[str, Tensor]:
    """Return a zero-filled copy of a block's state dict"""
    return {k: torch.zeros_like(v) for k, v in block_state_dict.items()}

def set_global_seed(seed: int) -> None:
    """Sets seed for all randomness sources."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_metrics(
        records: List[RecordDict], weighting_key: str
) -> MetricRecord:
    log_path = "experiments/results/metrics.json"
    
    # Run the original function
    aggregated = aggregate_metricrecords(records, weighting_key)
    
    data = dict(aggregated)
    is_training = any("train" in k for k in data.keys())

    history = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            pass

    if not history:
        current_round = 1
        new_record = {"round": current_round, **data}
        history.append(new_record)
    else:
        last_record = history[-1]

        has_train = any("train" in k for k in last_record.keys())
        has_eval = any("eval" in k or "val" in k for k in last_record.keys())

        if (is_training and not has_train) or (not is_training and not has_eval):
            # merge into last record
            last_record.update(data)
        else:
            # start new round
            current_round = last_record.get("round", 0) + 1
            new_record = {"round": current_round, **data}
            history.append(new_record)

    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)

    return aggregated


class NoMissingWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data_dict, label = self.base_dataset[idx]
        is_augmented = False
        return data_dict, label, is_augmented
    

class ZeroImputeWrapper(Dataset):
    """
    Fill-in the missing views with zero's
    """
    def __init__(self, base_dataset, missing_matrix):
        self.base_dataset = base_dataset
        self.missing_matrix = missing_matrix

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
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

    def __getitem__(self, idx):
        data_dict, label = self.base_dataset[idx]
        imputed_dict = {}

        # Check if sample has any missing values
        is_augmented = np.any(self.missing_matrix[:, idx])

        for v in range(self.missing_matrix.shape[0]):
            key = f"view_{v}"

            if self.missing_matrix[v, idx]:
                # Generate random Gaussian noise matching the shape of the missing view
                dim = data_dict[key].shape[0]
                raw_noise = torch.randn(dim)  # Gauss(mean 0, var 1)

                # L2 Normalize it so it matches the scale of the real views
                noise_tensor = torch.nn.functional.normalize(raw_noise, p=2, dim=0)

                # Apply intensity scaling
                imputed_dict[key] = noise_tensor * self.intensity
            else:
                imputed_dict[key] = data_dict[key]

        return imputed_dict, label, is_augmented

def get_missing_matrices(global_train, global_test, sensor_amount: int, missing_rate: float):
    """
    Creates matrices V x N -> True is missing, False is not missing.
    Guarantee at least 1 view per sample.
    """
    def generate_matrix(num_samples, seed):
        rng = np.random.default_rng(seed)

        # Drop some views independently based on missing rate (bool shape V x N)
        matrix = rng.random((sensor_amount, num_samples)) < missing_rate

        # Find samples (columns) where all views are missing
        all_missing_mask = np.all(matrix, axis=0)
        num_all_missing = np.sum(all_missing_mask)

        # If sample has no views, randomly add a view back
        if num_all_missing > 0:
            keep_indices = rng.integers(0, sensor_amount, size=num_all_missing)
            missing_sample_indices = np.where(all_missing_mask)[0]

            # Set that view to 'not missing'
            matrix[keep_indices, missing_sample_indices] = False

        return matrix

    # Generate both train and test matrices
    missing_matrix_train = generate_matrix(len(global_train), seed=42)
    missing_matrix_test = generate_matrix(len(global_test), seed=43)

    return missing_matrix_train, missing_matrix_test
