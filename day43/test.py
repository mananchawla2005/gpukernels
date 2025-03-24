import torch
import time
from gpu_kernels import flash_attention
import math

def reference_attention(q, k, v):
    d = q.size(1)
    # Compute scaled dot-product attention using PyTorch
    scores = torch.matmul(q, k.t()) / math.sqrt(d)
    attn = torch.softmax(scores, dim=1)
    out = torch.matmul(attn, v)
    return out

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # working with a size that complies with the kernel assumptions:
    # The kernel assumes d = 64, and uses tiling based on TILE_SIZE = 16.
    n = 64  # number of rows (queries/keys/values)
    d = 64  # embedding dimension
    
    # Create random input tensors on CPU (the binding copies them to the device).
    q = torch.rand((n, d), dtype=torch.float32)
    k = torch.rand((n, d), dtype=torch.float32)
    v = torch.rand((n, d), dtype=torch.float32)
    
    # Prepare an output tensor for flash_attention kernel result.
    o_flash = torch.zeros((n, d), dtype=torch.float32)
    
    # Run flash_attention kernel via binding (kernel does its own timing & prints its execution time)
    flash_attention(q, k, v, o_flash)
    
    # Compute reference attention on CPU
    o_ref = reference_attention(q, k, v)
    
    # Compare results
    diff = torch.norm(o_flash - o_ref)
    print("Difference between flash attention and reference:", diff.item())
    
    # Optionally, print outputs for further analysis (or visualize error)
    print("Flash Attention output:\n", o_flash)
    print("Reference output:\n", o_ref)