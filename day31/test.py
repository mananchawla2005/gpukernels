import torch
from gpu_kernels import tiled_matmul_corner_turned
import numpy as np

size = 4
M = torch.randn(size, size, dtype=torch.float32)
N = torch.randn(size, size, dtype=torch.float32)
P = torch.zeros(size, size, dtype=torch.float32)

expected = torch.mm(M, N)

tiled_matmul_corner_turned(M, N, P)

max_diff = torch.max(torch.abs(expected - P))
print(f"Maximum difference between CPU and GPU results: {max_diff}")

# Print matrices for small sizes
if size <= 4:
    print("\nMatrix M:")
    print(M)
    print("\nMatrix N:")
    print(N)
    print("\nExpected Result (CPU):")
    print(expected)
    print("\nGPU Result:")
    print(P)