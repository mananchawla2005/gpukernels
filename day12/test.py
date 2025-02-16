import torch
from gpu_kernels import layer_norm

# Test dimensions
batch_size = 32
hidden_size = 64

# Create input tensor on CPU (instead of GPU)
input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32)
output_tensor = torch.zeros_like(input_tensor)

# Calculate expected result using PyTorch's layer norm on CPU
expected = torch.nn.functional.layer_norm(input_tensor, (hidden_size,))

# Run our GPU implementation with CPU tensors
layer_norm(input_tensor, output_tensor)

# Compare results
max_diff = torch.max(torch.abs(expected - output_tensor))
print(f"Maximum difference between PyTorch CUDA and custom CUDA results: {max_diff}")

# Print sample of results
print("\nSample comparison (first 5 elements):")
print(f"PyTorch CUDA: {expected[0][:5]}")
print(f"Custom CUDA:  {output_tensor[0][:5]}")

# Verify layer norm properties
row_means = output_tensor.mean(dim=1)
row_stds = output_tensor.std(dim=1)
print("\nLayer Norm Properties Check:")
print(f"Row means (should be close to 0): {row_means.mean():.6f}")
print(f"Row stds (should be close to 1): {row_stds.mean():.6f}")