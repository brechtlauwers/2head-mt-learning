import torch
import os
import warnings

def setup_device():
    print("\n--- HARDWARE INFO ---")
    print(f"Node Name: {os.uname().nodename}")
    with open('/proc/cpuinfo', 'r') as f:
        for line in f:
            if 'model name' in line:
                print(f"CPU Model: {line.split(':')[1].strip()}")
                break
    print("---------------------\n")
    
    # Code to run this locally and avoid my old GPU
    device = torch.device("cpu")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
    
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 7:
            print(f"Compatible GPU found (Compute {major}.{minor}). Using CUDA.")
            device = torch.device("cuda:0")
        else:
            print(f"GPU found but too old (Compute {major}.{minor}).")

    return device

# Export the device so other files can just import it
DEVICE = setup_device()
