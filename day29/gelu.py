import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(input_ptr + offs, mask=mask)
    res = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(output_ptr + offs, res, mask=mask)

def gelu(input_h, output_h, n):
    input_t = torch.tensor(input_h, dtype=torch.float32, device='cuda')
    output_t = torch.empty((n,), dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    gelu_kernel[grid](input_t, output_t, n, BLOCK_SIZE=BLOCK_SIZE)
    output_h[:] = output_t.cpu().numpy()