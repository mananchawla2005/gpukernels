import torch
import time
from gpu_kernels import dynamic_tiled_matmul

# Setup
size = 512
M = torch.randn(size, size, dtype=torch.float32)
N = torch.randn(size, size, dtype=torch.float32)
P = torch.zeros(size, size, dtype=torch.float32)

# Warmup runs
for _ in range(3):
    dynamic_tiled_matmul(M, N, P)
    _ = torch.mm(M.cuda(), N.cuda())
torch.cuda.synchronize()

iterations = 5

# CUDA kernel benchmark
cuda_times = []
for _ in range(iterations):
    torch.cuda.synchronize()
    start_cuda_time = time.perf_counter()
    dynamic_tiled_matmul(M, N, P)
    torch.cuda.synchronize()
    end_cuda_time = time.perf_counter()
    cuda_times.append(end_cuda_time - start_cuda_time)

print(f"\nCUDA KERNEL BENCHMARK TIME (avg): {sum(cuda_times)/iterations}")
print(cuda_times)
# PyTorch GPU benchmark
M_gpu = M.cuda()
N_gpu = N.cuda()
torch_times = []
for _ in range(iterations):
    torch.cuda.synchronize()
    start_torch_time = time.perf_counter()
    P_gpu = torch.mm(M_gpu, N_gpu)
    torch.cuda.synchronize()
    end_torch_time = time.perf_counter()
    torch_times.append(end_torch_time - start_torch_time)

print(f'PYTORCH GPU BENCHMARK TIME (avg): {sum(torch_times)/iterations}')

# Verify results
cuda_result = P.clone()
dynamic_tiled_matmul(M, N, cuda_result)
torch_result = P_gpu.cpu()
max_diff = torch.max(torch.abs(cuda_result - torch_result))
print(f"\nMaximum difference between CUDA and PyTorch results: {max_diff}")