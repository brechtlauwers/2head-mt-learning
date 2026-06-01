import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

from core.datasets import (
    ModelNet40Features, VehicleSensITFeatures, UCIHARFeatures,
    YoutubeFeatures
)
from core.data import (
    NoMissingWrapper, ZeroImputeWrapper, GausImputeWrapper, generate_missing_matrix,
    CPMImputeWrapper, CPMIndexDataset, DCPImputeWrapper
)

from core.cpm_nets import CPMNet_Works
from core.dcp_nets import DCP_Net, dcp_loss

from centralized.config import DEVICE
from centralized.utils import get_dataset_details


def get_dataloaders(dataset_name: str, experiment_type: str, missing_rate: float,
                    seed: int, batch_size=32, use_validation: bool = True):
    """Return the data loaders and the sensor dimensions"""
    # Load the dataset
    if dataset_name == "modelnet":
        base_path = "data/features/ModelNet40_mvcnn_gvcnn.mat"
        global_train = ModelNet40Features(features_path=base_path, split="train")
        global_test = ModelNet40Features(features_path=base_path, split="test")
        sensor_dims = {0: 4096, 1: 2048}
    elif dataset_name == "sensit":
        base_path = "data/features/sensit_vehicle_features/"
        global_train = VehicleSensITFeatures(root_dir=base_path, split="train")
        global_test = VehicleSensITFeatures(root_dir=base_path, split="test")
        sensor_dims = {0: 50, 1: 50}
    elif dataset_name == "har":
        base_path = "data/features/uci_har_dataset"
        global_train = UCIHARFeatures(root_dir=base_path, split="train")
        global_test = UCIHARFeatures(root_dir=base_path, split="test")
        sensor_dims = {0: 348, 1: 213}
    elif dataset_name == "youtube":
        base_path = "data/features/youtube"
        sensor_dims = {0: 1000, 1: 512, 2: 2000}
        global_train = YoutubeFeatures(root_dir=base_path, split="train", sensor_dims=sensor_dims)
        global_test = YoutubeFeatures(root_dir=base_path, split="test", sensor_dims=sensor_dims)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found.")

    if use_validation:
        val_size = int(0.15 * len(global_train))
        train_size = len(global_train) - val_size

        global_train, global_val = random_split(
            global_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        global_val = None

    train_len = len(global_train)
    val_len = len(global_val) if global_val is not None else 0
    test_len = len(global_test)
        
    num_sensors = len(sensor_dims)
    active_sensors = list(range(num_sensors))  # e.g. [0, 1] for 2 sensors

    # Generate missing matrices
    missing_matrix_train = generate_missing_matrix(
        train_len, num_sensors, active_sensors, missing_rate, seed=seed
    )

    if use_validation:
        missing_matrix_val = generate_missing_matrix(
            val_len, num_sensors, active_sensors, missing_rate, seed=seed+2
        )
    else:
        missing_matrix_val = None
    
    missing_matrix_test = generate_missing_matrix(
        test_len, num_sensors, active_sensors, missing_rate, seed=seed+1
    )

    val_ds = None

    # Load the correct experiment train and test dataset
    if "baseline" in experiment_type:
        # No missing data, use the whole dataset
        train_ds = NoMissingWrapper(global_train)
        test_ds = NoMissingWrapper(global_test)
        if use_validation and global_val is not None:
            val_ds = NoMissingWrapper(global_val)

    elif experiment_type == "drop_missing":
        # Find indices of samples with at least one missing view
        complete_train_mask = ~np.any(missing_matrix_train, axis=0)
        complete_test_mask = ~np.any(missing_matrix_test, axis=0)
        train_indices = np.where(complete_train_mask)[0].tolist()
        test_indices = np.where(complete_test_mask)[0].tolist()

        print(f"[{dataset_name}] Remaining train samples: {len(train_indices)} ({len(train_indices)/len(global_train)*100:.1f}%)")
        print(f"[{dataset_name}] Remaining test samples: {len(test_indices)} ({len(test_indices)/len(global_test)*100:.1f}%)")

        train_ds = NoMissingWrapper(Subset(global_train, train_indices))
        test_ds = NoMissingWrapper(Subset(global_test, test_indices))

        if global_val is not None and missing_matrix_val is not None:
            complete_val_mask = ~np.any(missing_matrix_val, axis=0)
            val_indices = np.where(complete_val_mask)[0].tolist()
            val_ds = NoMissingWrapper(Subset(global_val, val_indices))
        
    elif "impute_zero" in experiment_type:
        total_missing_train = np.sum(missing_matrix_train)
        total_missing_test = np.sum(missing_matrix_test)
        
        print(f"[{dataset_name}] Naive impute (train): Replaced {total_missing_train}/{missing_matrix_train.size} views with zeros.")
        print(f"[{dataset_name}] Naive impute (test): Replaced {total_missing_test}/{missing_matrix_test.size} views with zeros.")

        # Wrap both datasets to inject zeros
        train_ds = ZeroImputeWrapper(global_train, missing_matrix_train)
        test_ds = ZeroImputeWrapper(global_test, missing_matrix_test)
        if use_validation:
            val_ds = ZeroImputeWrapper(global_val, missing_matrix_val)

    elif "impute_gaus" in experiment_type:
        total_missing_train = np.sum(missing_matrix_train)
        total_missing_test = np.sum(missing_matrix_test)
        
        print(f"[{dataset_name}] Gen impute (train): Replaced {total_missing_train}/{missing_matrix_train.size} views with noise.")
        print(f"[{dataset_name}] Gen impute (test): Replaced {total_missing_test}/{missing_matrix_test.size} views with noise.")

        # Wrap both datasets to inject normalized Gaussian noise
        train_ds = GausImputeWrapper(global_train, missing_matrix_train, intensity=1.0)
        test_ds = GausImputeWrapper(global_test, missing_matrix_test, intensity=1.0)
        if use_validation:
            val_ds = GausImputeWrapper(global_val, missing_matrix_val, intensity=1.0)

    elif "impute_cpm" in experiment_type:
        # Use CPM-nets as generative method
        # https://github.com/tbh-98/Reproducing-of-CPM-Nets-Cross-Partial-Multi-View-Networks
        num_classes, hidden_dim, _, cpm_epochs, _, _, _ = get_dataset_details(dataset_name)

        val_tag = "val" if use_validation else "full"
        weights_file = (
            f"./weights/cpm_weights_{dataset_name}_missing{missing_rate}"
            f"_seed{seed}_{val_tag}.pth"
        )

        print(weights_file)

        cpm_model = CPMNet_Works(
            device=DEVICE,
            view_dims=sensor_dims,
            trainLen=len(global_train),
            testLen=len(global_test),
            lsd_dim=hidden_dim
        ).to(DEVICE)

        if os.path.exists(weights_file):
            print(f"[{dataset_name}] Found cached cpm-net. Loading weights from '{weights_file}'...")
            # Load the saved model state directly
            cpm_model.load_state_dict(torch.load(weights_file, map_location=DEVICE, weights_only=True))
        else:
            print(f"[{dataset_name}] Training cpm-net from scratch...")
            
            # Train cpm phase
            train_loader_cpm = DataLoader(
                CPMIndexDataset(global_train, missing_matrix_train),
                batch_size=128, shuffle=True
            )
            cpm_model.train_model(train_loader_cpm, num_classes=num_classes, epoch=cpm_epochs)

            # Test cpm phase
            test_loader_cpm = DataLoader(
                CPMIndexDataset(global_test, missing_matrix_test),
                batch_size=128, shuffle=False
            )
            cpm_model.test_model(test_loader_cpm, epoch=cpm_epochs)
            
            # Save the trained weights to disk
            os.makedirs("./weights", exist_ok=True)
            torch.save(cpm_model.state_dict(), weights_file)
            print(f"[{dataset_name}] cpm-net weights successfully saved to '{weights_file}'")

        cpm_model.eval()

        total_missing_train = np.sum(missing_matrix_train)
        print(f"[{dataset_name}] CPM-Nets: Synthesizing {total_missing_train} missing views based on learned latent space.")

        train_ds = CPMImputeWrapper(global_train, missing_matrix_train, cpm_model, mode="train")
        test_ds = CPMImputeWrapper(global_test, missing_matrix_test, cpm_model, mode="test")
        if use_validation and global_val is not None:
            val_h_file = (
                f"./weights/cpm_val_h_{dataset_name}_missing{missing_rate}"
                f"_seed{seed}.pth"
            )

            cpm_val = CPMNet_Works(
                device=DEVICE,
                view_dims=sensor_dims,
                trainLen=1,  # placeholder
                testLen=len(global_val),
                lsd_dim=hidden_dim
            ).to(DEVICE)

            # Copy trained decoders
            cpm_val.net.load_state_dict(cpm_model.net.state_dict())
            
            if os.path.exists(val_h_file):
                print(f"[{dataset_name}] Loading cached validation latents...")
                saved = torch.load(val_h_file, map_location=DEVICE, weights_only=True)
                cpm_val.h_test.data.copy_(saved["h_val"])
            else:
                print(f"[{dataset_name}] Optimizing CPM latents for validation set...")
                val_loader_cpm = DataLoader(
                    CPMIndexDataset(global_val, missing_matrix_val),
                    batch_size=128, shuffle=False
                )
                # Optimizes only cpm_val.h_test; decoders set to eval internally
                cpm_val.test_model(val_loader_cpm, epoch=cpm_epochs)

                os.makedirs("./weights", exist_ok=True)
                torch.save({"h_val": cpm_val.h_test.data}, val_h_file)

            cpm_val.eval()
            val_ds = CPMImputeWrapper(
                global_val, missing_matrix_val, cpm_val, mode="test"
            )

    elif "impute_dcp" in experiment_type:
        # Use DCP as generative method
        # https://github.com/XLearning-SCU/2022-TPAMI-DCP

        num_classes, hidden_dim, _, _, dcp_epochs, _, _ = get_dataset_details(dataset_name)

        dcp_model = DCP_Net(
            view_dims=sensor_dims, 
            latent_dim=128, 
            hidden_dim=hidden_dim
        ).to(DEVICE)

        weights_file = f"./weights/dcp_weights_{dataset_name}_missing{missing_rate}_seed{seed}.pth"

        if os.path.exists(weights_file):
            print(f"[{dataset_name}] Found cached dcp-net. Loading weights from '{weights_file}'...")
            # Load the saved model state directly
            dcp_model.load_state_dict(torch.load(weights_file, map_location=DEVICE, weights_only=True))
        else:
            print(f"[{dataset_name}] No cache found. Training dcp-net from scratch...")
            masked_train_ds = CPMIndexDataset(global_train, missing_matrix_train)
            train_loader_dcp = DataLoader(masked_train_ds, batch_size=128, shuffle=True)

            optimizer = torch.optim.Adam(dcp_model.parameters(), lr=0.001)
            dcp_model.train()

            for epoch in range(dcp_epochs):
                total_loss = 0.0
                for _, inputs_dict, _, sn_mask in train_loader_dcp:
                    inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
                    sn_mask = {k: v.to(DEVICE) for k, v in sn_mask.items()}

                    optimizer.zero_grad()

                    loss = dcp_loss(dcp_model, inputs_dict, sn_mask)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"DCP Pre-train Epoch {epoch+1}/{dcp_epochs} | Loss: {total_loss/len(train_loader_dcp):.4f}")

            # Save the trained weights to disk
            os.makedirs("./weights", exist_ok=True)
            torch.save(dcp_model.state_dict(), weights_file)
            print(f"[{dataset_name}] DCP-net weights successfully saved to '{weights_file}'")

        dcp_model.eval()

        total_missing_train = np.sum(missing_matrix_train)
        print(f"[{dataset_name}] DCP: Synthesizing {total_missing_train} missing views via Contrastive Latent Translation.")

        train_ds = DCPImputeWrapper(global_train, missing_matrix_train, dcp_model)
        test_ds = DCPImputeWrapper(global_test, missing_matrix_test, dcp_model)
        if use_validation:
            val_ds = DCPImputeWrapper(global_val, missing_matrix_val, dcp_model)

    else:
        raise ValueError("Unknown experiment type.")

    # Create data loaders
    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),  # Optimization for cloud
    )
    testloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),  # Optimization for cloud
    )

    valloader = None
    if use_validation and val_ds is not None:
        valloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(DEVICE.type == "cuda"),  # Optimization for cloud
            num_workers=0  # Optimization for cloud
        )

    return trainloader, valloader, testloader, sensor_dims
