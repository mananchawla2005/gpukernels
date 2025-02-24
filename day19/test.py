import torch
import os
import math

# Add torch lib path for dependency DLLs
print(torch.__file__)
dll_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.add_dll_directory(dll_path)

import gpu_kernels

# Create input tensor (Q, K, V)
x = torch.randn(32, 64, dtype=torch.float32)  # batch_size=32, embedding_dim=64

# Prepare output tensor for custom GPU kernel
y_custom = torch.zeros_like(x)
d = 64  # scaling factor

# Compute self attention using your GPU kernel
gpu_kernels.self_attention(x, y_custom, d)

# Compute self attention purely in PyTorch:
# 1. Compute scores: Q * K^T / sqrt(d) (note: here Q, K, V are all x)
scores = torch.matmul(x, x.transpose(0, 1)) / math.sqrt(d)
# 2. Apply softmax across dim=-1
soft = torch.softmax(scores, dim=-1)
# 3. Multiply with V (x)
y_ref = torch.matmul(soft, x)

# Compare results
diff = torch.abs(y_custom - y_ref).max().item()
print("Custom GPU self_attention output:")
print(y_custom)
print("Reference PyTorch self_attention output:")
print(y_ref)
print(f"Maximum difference: {diff:.6f}")