import torch
from gpu_kernels import softmax

# Test dimensions
batch_size = 32
hidden_size = 64

# Create input tensor on CPU (instead of GPU)
input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32)
output_tensor = torch.zeros_like(input_tensor)

# Calculate expected result using PyTorch's layer norm on CPU
expected = torch.nn.functional.softmax(input_tensor, dim=-1)

# Run our GPU implementation with CPU tensors
softmax(input_tensor, output_tensor)

# Compare results
max_diff = torch.max(torch.abs(expected - output_tensor))
print(f"Maximum difference between PyTorch CUDA and custom CUDA results: {max_diff}")

# Print sample of results
# print("\nSample comparison (first 5 elements):")
print(f"PyTorch CUDA: {expected}")
print(f"Custom CUDA:  {output_tensor}")
