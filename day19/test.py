from gpu_kernels import self_attention
import torch

# Create input tensor
x = torch.randn(32, 64, dtype=torch.float32)  # batch_size=32, embedding_dim=64
d = 64  # scaling factor

# Compute self attention
output = self_attention(x, d)