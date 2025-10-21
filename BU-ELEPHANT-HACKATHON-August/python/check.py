import torch

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Example: move a tensor to the device
x = torch.rand(3, 3).to(device)
print(f"Tensor device: {x.device}")
