import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device index: {torch.cuda.current_device()}")
    # Make sure to get device properties using the index if device_count > 0
    if torch.cuda.device_count() > 0:
        # Get properties of the current device
        current_device_idx = torch.cuda.current_device()
        print(f"Device name: {torch.cuda.get_device_name(current_device_idx)}")
        print(f"Device capability: {torch.cuda.get_device_capability(current_device_idx)}")
        total_memory = torch.cuda.get_device_properties(current_device_idx).total_memory
        print(f"Total GPU Memory: {total_memory / (1024**2):.2f} MB") # Convert bytes to MB
    else:
        print("CUDA device count is 0, cannot get device name.")
else:
    print("CUDA not available. PyTorch is likely using CPU or another backend if configured (e.g., MPS for Apple Silicon).")
    # Check for MPS (Apple Silicon) as an alternative
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) is available.")
    else:
        print("MPS (Apple Silicon GPU) is not available.")
