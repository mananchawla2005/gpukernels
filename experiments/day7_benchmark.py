import torch
import time
from gpu_kernels import gelu

# Setup
size = 1000000  # Vector size for GELU
input_tensor = torch.randn(1, size, dtype=torch.float32).cuda()
output_tensor = torch.zeros_like(input_tensor)

# Warmup runs
for _ in range(3):
    gelu(input_tensor, output_tensor)
    _ = torch.nn.functional.gelu(input_tensor)
torch.cuda.synchronize()

iterations = 5

# CUDA kernel benchmark
cuda_times = []
for _ in range(iterations):
    torch.cuda.synchronize()
    start_cuda_time = time.perf_counter()
    gelu(input_tensor, output_tensor)
    torch.cuda.synchronize()
    end_cuda_time = time.perf_counter()
    cuda_times.append(end_cuda_time - start_cuda_time)

print(f"\nCUDA KERNEL BENCHMARK TIME (avg): {sum(cuda_times)/iterations}")

# PyTorch GPU benchmark
torch_times = []
for _ in range(iterations):
    torch.cuda.synchronize()
    start_torch_time = time.perf_counter()
    torch_result = torch.nn.functional.gelu(input_tensor)
    torch.cuda.synchronize()
    end_torch_time = time.perf_counter()
    torch_times.append(end_torch_time - start_torch_time)

print(f'PYTORCH GPU BENCHMARK TIME (avg): {sum(torch_times)/iterations}')

# Verify results
cuda_result = output_tensor.clone()
gelu(input_tensor, cuda_result)
torch_result = torch.nn.functional.gelu(input_tensor)
max_diff = torch.max(torch.abs(cuda_result - torch_result))
print(f"\nMaximum difference between CUDA and PyTorch results: {max_diff}")