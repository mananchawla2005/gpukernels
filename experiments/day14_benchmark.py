import torch
import time
from gpu_kernels import softmax

# Test dimensions
batch_size = 1024
hidden_size = 1024

# Create input tensor on CPU
input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32)
output_tensor = torch.zeros_like(input_tensor)

# Time PyTorch CPU implementation
start_time = time.perf_counter()
expected_cpu = torch.nn.functional.softmax(input_tensor, dim=-1)
cpu_time = (time.perf_counter() - start_time) * 1000
print(f"PyTorch CPU time: {cpu_time:.3f} ms")

# Time PyTorch GPU implementation
if torch.cuda.is_available():
    input_gpu = input_tensor.cuda()
    _ = torch.nn.functional.softmax(input_gpu, dim=-1)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    expected_gpu = torch.nn.functional.softmax(input_gpu, dim=-1)
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start_time) * 1000
    print(f"PyTorch GPU time: {gpu_time:.3f} ms")
    expected_gpu = expected_gpu.cpu()  # Move back to CPU for comparison

# Time our custom CUDA implementation
start_time = time.perf_counter()
softmax(input_tensor, output_tensor)
custom_time = (time.perf_counter() - start_time) * 1000
print(f"Custom CUDA time: {custom_time:.3f} ms")

# Compare results with CPU PyTorch
max_diff_cpu = torch.max(torch.abs(expected_cpu - output_tensor))
print(f"\nMaximum difference between PyTorch CPU and custom CUDA: {max_diff_cpu}")

if torch.cuda.is_available():
    max_diff_gpu = torch.max(torch.abs(expected_gpu - output_tensor))
    print(f"Maximum difference between PyTorch GPU and custom CUDA: {max_diff_gpu}")

# Print sample of results
print("\nSample comparison (first row):")
print(f"PyTorch CPU:   {expected_cpu[0]}")
if torch.cuda.is_available():
    print(f"PyTorch GPU:   {expected_gpu[0]}")
print(f"Custom CUDA:   {output_tensor[0]}")