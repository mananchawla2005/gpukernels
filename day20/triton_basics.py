import torch
import triton
import triton.language as tl
import os

def cdiv(a,b): return (a + b - 1) // b

def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"

def tensor_copy(x, bs, fn):
    z = torch.zeros_like(x)
    check_tensors_gpu_ready(x, z)
    n = x.numel()
    n_blocks = cdiv(n, bs)
    grid = (n_blocks, )
    fn[grid](x, z, n, bs)
    return z

@triton.jit
def tensor_copy_kernel(x_d, z_d, n, bs: tl.constexpr):
    block_id = tl.program_id(0)
    offs = block_id*bs + tl.arange(0, bs)
    mask = offs < n
    x = tl.load(x_d+offs, mask) # Only if mask vector is True else undefined
    tl.store(z_d+offs, x, mask)

x = torch.tensor([1,2,3,4,5,6]).to('cuda')
z = tensor_copy(x, bs=2, fn=tensor_copy_kernel)
print(x, z)


@triton.jit
def matmul_kernel(a_d, b_d, c_d, m, n, k, stride_a0, stride_a1, stride_b0, stride_b1, stride_c0, stride_c1, bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr ):
    block_idx = tl.program_id(0)
    block_idy = tl.program_id(1)
    rm = bm * block_idx + tl.arange(0, bm)
    rn = bn * block_idy + tl.arange(0, bn)
    rk = tl.arange(0, bk)
    offs_a = a_d + (rm[:, None] * stride_a0 + rk[None, :] * stride_a1)
    offs_b = b_d + (rk[:, None] * stride_b0 + rn[None, :] * stride_b1)

    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for _ in range(0, k, bk):
        a = tl.load(offs_a, mask=rm[:, None] < m, other=0.0)
        b = tl.load(offs_b, mask=rn[None, :] < n, other=0.0)
        acc += tl.dot(a, b)
        offs_a+=bk*stride_a1
        offs_b+=bk*stride_b0
    c = c_d+(rm[:, None] * stride_c0 + rn[None, :] * stride_c1)
    mask = (rm[:, None] < m) & (rn[None, :] < n)
    tl.store(c, acc, mask)


def matmul(a: torch.Tensor, b: torch.Tensor, block_size_m: int = 16, block_size_n: int = 16, block_size_k: int = 16):
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    assert a.size(1) == b.size(0), f"Incompatible dimensions: {a.size()} x {b.size()}"
    check_tensors_gpu_ready(a, b)
    
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    grid = (cdiv(M, block_size_m), cdiv(N, block_size_n))
    
    matmul_kernel[grid](
        a_d=a, b_d=b, c_d=c,
        m=M, n=N, k=K,
        stride_a0=stride_am, stride_a1=stride_ak,
        stride_b0=stride_bk, stride_b1=stride_bn,
        stride_c0=stride_cm, stride_c1=stride_cn,
        bm=block_size_m, bn=block_size_n, bk=block_size_k
    )
    
    return c

if __name__ == "__main__":
    a = torch.randn(128, 256, device='cuda')
    b = torch.randn(256, 64, device='cuda')
    
    c_triton = matmul(a, b)
    print(c_triton)
    # Verify against PyTorch
    c_torch = torch.matmul(a, b)
    print(c_torch)
assert torch.allclose(c_triton, c_torch, rtol=5e-2, atol=5e-2), "Results don't match!"
print("Matrix multiplication test passed!")