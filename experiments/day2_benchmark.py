import torch
import time
from gpu_kernels import vec_add

# CUDA benchmark
A = torch.ones(1_000_000)
B = torch.ones_like(A)
C = torch.zeros_like(A)

iterations = 5
cuda_times = []
for _ in range(iterations):
    start_cuda_time = time.perf_counter()
    vec_add(A, B, C)
    torch.cuda.synchronize()
    end_cuda_time = time.perf_counter()
    cuda_times.append(end_cuda_time - start_cuda_time)

print(f"\nCUDA BENCHMARK TIME (avg): {sum(cuda_times)/iterations}")

# Naive benchmark
naive_times = []
for _ in range(iterations):
    start_naive_time = time.perf_counter()
    for i in range(1_000_000):
        C[i] = A[i] + B[i]
    end_naive_time = time.perf_counter()
    naive_times.append(end_naive_time - start_naive_time)

print(f'NAIVE BENCHMARK TIME (avg): {sum(naive_times)/iterations}')

# PyTorch benchmark
torch_times = []
for _ in range(iterations):
    start_torch_time = time.perf_counter()
    C = torch.add(A, B)
    end_torch_time = time.perf_counter()
    torch_times.append(end_torch_time - start_torch_time)

print(f'PYTORCH BENCHMARK TIME (avg): {sum(torch_times)/iterations}')