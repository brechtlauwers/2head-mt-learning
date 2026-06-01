import os
import gzip
from typing import Optional, Tuple, Dict
import numpy as np
import pickle
import scipy.io as scio
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_svmlight_file

"""
Functions to load specific datasets or features from disk into a PyTorch Dataset class.
"""

class ModelNet40Features(Dataset):
    '''
    Extract the features of ModelNet40 dataset.
    - View 1 (MVCNN): 4096 dimensions
    - View 2 (GVCNN): 2048 dimensions
    '''
    def __init__(
            self, features_path: str, split: str = 'all',
            transform: Optional[bool] = None
    ):
        self.transform = transform

        # Load the features file
        try:
            data = scio.loadmat(features_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"features file not found at {features_path}")

        # Process labels
        lbls = data['Y'].astype(np.int64).squeeze()
        # Adjust matlab indexing (1) to python indexing (0)
        if lbls.min() == 1:
            lbls = lbls - 1

        all_indices = data['indices'].item().squeeze()

        # Extract the features
        view0 = data['X'][0].item().astype(np.float32)
        view1 = data['X'][1].item().astype(np.float32)

        if split == 'train':
            mask = (all_indices == 1)
        elif split == 'test':
            mask = (all_indices == 0)
        else:  # 'all'
            mask = np.ones_like(all_indices, dtype=bool)

        self.labels = lbls[mask]
        self.view0 = view0[mask]
        self.view1 = view1[mask]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        # Get raw features
        v0 = torch.from_numpy(self.view0[idx])
        v1 = torch.from_numpy(self.view1[idx])
        label = self.labels[idx]

        # L2 normalization
        v0 = torch.nn.functional.normalize(v0, p=2, dim=0)
        v1 = torch.nn.functional.normalize(v1, p=2, dim=0)

        data_dict = {
            "view_0": v0,
            "view_1": v1
        }

        return data_dict, label


class VehicleSensITFeatures(Dataset):
    '''
    Extract the features of the vehicle sensor dataset (sensIT)
    - View 0 (Acoustic): 50 dimensions
    - View 1 (Seismic): 50 dimensions
    '''
    def __init__(self, root_dir: str, split: str = 'all', transform: Optional[bool] = None):
        self.transform = transform

        # Load features
        train_files = {
            'view0': os.path.join(root_dir, 'acoustic'),
            'view1': os.path.join(root_dir, 'seismic')
        }
        test_files = {
            'view0': os.path.join(root_dir, 'acoustic.t'),
            'view1': os.path.join(root_dir, 'seismic.t')
        }

        # Determine which data to load
        files_to_load = []
        if split == 'train':
            files_to_load.append(train_files)
        elif split == 'test':
            files_to_load.append(test_files)
        else:  # 'all'
            files_to_load.append(train_files)
            files_to_load.append(test_files)

        v0_list = []
        v1_list = []
        labels_list = []

        # Load the data
        for f_dict in files_to_load:
            # Check if they exist
            if not os.path.exists(f_dict['view0']) or not os.path.exists(f_dict['view1']):
                raise FileNotFoundError(f"Missing data files in {root_dir}")

            # Load View 0, acoustic
            data_v0, y_v0 = load_svmlight_file(f_dict['view0'], n_features=50)
            data_v0 = data_v0.toarray().astype(np.float32)

            # Load View 1, Seismic
            data_v1, _ = load_svmlight_file(f_dict['view1'], n_features=50)
            data_v1 = data_v1.toarray().astype(np.float32)

            v0_list.append(data_v0)
            v1_list.append(data_v1)
            labels_list.append(y_v0)

        self.view0 = np.concatenate(v0_list, axis=0)
        self.view1 = np.concatenate(v1_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)

        # convert to long for pytorch
        self.labels = self.labels.astype(np.int64)

        # Keep only classes 1 and 2 (AAV and DW)
        mask = (self.labels != 3)
        
        self.view0 = self.view0[mask]
        self.view1 = self.view1[mask]
        self.labels = self.labels[mask]

        # labels start at 1, so convert to 0
        if self.labels.min() == 1:
            self.labels = self.labels - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        # Get raw features
        v0 = torch.from_numpy(self.view0[idx])
        v1 = torch.from_numpy(self.view1[idx])
        label = self.labels[idx]

        # L2 normalization
        v0 = torch.nn.functional.normalize(v0, p=2, dim=0)
        v1 = torch.nn.functional.normalize(v1, p=2, dim=0)

        data_dict = {
            "view_0": v0,
            "view_1": v1
        }

        return data_dict, label


class UCIHARFeatures(Dataset):
    """
    Extract the features of the UCI HAR dataset
    - View 0: Accelerometer (348 dim)
    - View 1: Gyroscope (213 dim)
    """
    def __init__(self, root_dir: str, split: str = 'all'):
        self.split = split

        features_path = os.path.join(root_dir, "features.txt")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"features not found in {root_dir}")

        with open(features_path, 'r') as f:
            # line example "1 tBodyAcc-mean()-X"
            feature_names = [line.strip().split()[1] for line in f.readlines()]

        # Slice into two modalities
        acc_idx = []
        gyro_idx = []
        for i, name in enumerate(feature_names):
            if 'Gyro' in name:
                gyro_idx.append(i)
            else:
                # Accelerometer + 3 extra gravity features
                acc_idx.append(i)

        # Determine which splits to load
        splits_to_load = []
        if split == 'train':
            splits_to_load.append('train')
        elif split == 'test':
            splits_to_load.append('test')
        else:  # 'all'
            splits_to_load.extend(['train', 'test'])

        x_list = []
        y_list = []

        # Load the data
        for sp in splits_to_load:
            x_path = os.path.join(root_dir, sp, f"X_{sp}.txt")
            y_path = os.path.join(root_dir, sp, f"y_{sp}.txt")

            if not os.path.exists(x_path) or not os.path.exists(y_path):
                raise FileNotFoundError(f"Missing data files for split '{sp}' in {root_dir}")

            # Load normalized features and labels
            x_data = np.loadtxt(x_path, dtype=np.float32)
            y_data = np.loadtxt(y_path, dtype=np.int64)

            x_list.append(x_data)
            y_list.append(y_data)

        # Concatenate both splits if 'all'
        X_full = np.concatenate(x_list, axis=0)
        self.labels = np.concatenate(y_list, axis=0).squeeze()

        # Labels are 1-6, convert to 0-5
        if self.labels.min() == 1:
            self.labels = self.labels - 1

        # Split unified array into V0 and V1
        self.view0 = X_full[:, acc_idx]
        self.view1 = X_full[:, gyro_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], np.int64]:
        # Get raw features
        v0 = torch.from_numpy(self.view0[idx])
        v1 = torch.from_numpy(self.view1[idx])
        label = self.labels[idx]

        # L2 normalization
        v0 = torch.nn.functional.normalize(v0, p=2, dim=0)
        v1 = torch.nn.functional.normalize(v1, p=2, dim=0)

        data_dict = {
            "view_0": v0,
            "view_1": v1
        }

        return data_dict, label


class YoutubeFeatures(Dataset):
    """
    Extracted features from the YouTube multiview video games dataset.
    - View 0: Text (1000 dim)
    - View 1: Vision (512 dim)
    - View 2: Audio (2000 dim)
    """
    def __init__(self, root_dir: str, split: str, sensor_dims: dict[int, int]):
        self.split = split
        self.sensor_dims = sensor_dims

        # Specific feature files for the 3 views
        self.file_names = {
            0: "text_game_lda_1000.txt",   # Very strong text features
            1: "vision_cuboids_histogram.txt",      # Medium vision features
            2: "audio_mfcc.txt"           # Weak audio features
        }

        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found at {split_dir}")

        # Temp hold parsed data by instance ID
        parsed_views = {0: {}, 1: {}, 2: {}}
        labels_dict = {}

        # Parse the files
        for view_idx, file_name in self.file_names.items():
            file_path = os.path.join(split_dir, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Feature file missing: {file_path}")

            max_dim = self.sensor_dims[view_idx]
            current_id = None

            with open(file_path, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if not line:   # Skip empty lines
                        continue

                    # Track instance ID
                    if line.startswith('#I'):
                        current_id = int(line.split()[1])

                    # Parse feature vector
                    elif not line.startswith('#'):
                        parts = line.split()

                        # The first number is the class label 1-31 so convert to 0-30
                        label = int(parts[0]) - 1
                        labels_dict[current_id] = label
                        
                        # Create array for this view
                        features = np.zeros(max_dim, dtype=np.float32)
                        
                        # Parse the id:value pairs
                        for pair in parts[1:]:
                            fid_str, fval_str = pair.split(':')
                            fid = int(fid_str) - 1 # 1-indexed to 0-indexed
                            
                            # Safety check to prevent dimension crashes
                            if fid < max_dim:
                                features[fid] = float(fval_str)
                        
                        parsed_views[view_idx][current_id] = features

        # Align the modalities
        # Only keep videos that have all 3 views (so intersection)
        valid_ids = set(parsed_views[0].keys())
        valid_ids = valid_ids.intersection(parsed_views[1].keys())
        valid_ids = valid_ids.intersection(parsed_views[2].keys())
        
        valid_ids = sorted(list(valid_ids))

        # Build the final tensors
        v0_list, v1_list, v2_list, labels_list = [], [], [], []
        
        for vid in valid_ids:
            v0_list.append(parsed_views[0][vid])
            v1_list.append(parsed_views[1][vid])
            v2_list.append(parsed_views[2][vid])
            labels_list.append(labels_dict[vid])

        self.view0 = np.stack(v0_list, axis=0)
        self.view1 = np.stack(v1_list, axis=0)
        self.view2 = np.stack(v2_list, axis=0)
        self.labels = np.array(labels_list, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], np.int64]:
        v0 = torch.from_numpy(self.view0[idx])
        v1 = torch.from_numpy(self.view1[idx])
        v2 = torch.from_numpy(self.view2[idx])
        label = self.labels[idx]

        # L2 normalization
        v0 = torch.nn.functional.normalize(v0, p=2, dim=0)
        v1 = torch.nn.functional.normalize(v1, p=2, dim=0)
        v2 = torch.nn.functional.normalize(v2, p=2, dim=0)

        data_dict = {
            "view_0": v0,
            "view_1": v1,
            "view_2": v2
        }

        return data_dict, label
