import torch
from gpu_kernels import vec_add

A = torch.ones(1000)
B = torch.ones(1000)
C = torch.zeros_like(A)

vec_add(A, B, C)
print(C)