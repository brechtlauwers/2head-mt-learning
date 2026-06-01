"""Simple helper functions."""


def count_params(model):
    """Return total trainable parameters (in millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_dataset_details(dataset_name: str):
    """Return details for a given dataset"""
    num_classes, hidden_dim, batch_size, cpm_epochs, dcp_epochs, lmbda, lr = 0, 0, 0, 0, 0, 0, 0
    
    if dataset_name == "modelnet":
        num_classes = 40
        hidden_dim = 128
        batch_size = 32
        cpm_epochs = 50
        dcp_epochs = 100
        lmbda = 3
        lr = 0.001
    elif dataset_name == "sensit":
        num_classes = 2
        hidden_dim = 64
        batch_size = 32
        cpm_epochs = 20
        dcp_epochs = 100
        lmbda = 1
        lr = 0.001
    elif dataset_name == "har":
        num_classes = 6
        hidden_dim = 64
        batch_size = 32
        cpm_epochs = 20
        dcp_epochs = 100
        lmbda = 2
        lr = 0.001
    elif dataset_name == "youtube":
        num_classes = 31
        hidden_dim = 128
        batch_size = 256
        cpm_epochs = 50
        dcp_epochs = 100
        lmbda = 2
        lr = 0.001
    else:
        raise ValueError(f"Dataset {dataset_name} does not exist!")

    return num_classes, hidden_dim, batch_size, cpm_epochs, dcp_epochs, lmbda, lr
