import triton
import triton.language as tl
import torch
from torch.testing import assert_close

@triton.jit
def silu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)          
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    output = x * sigmoid_x
    tl.store(output_ptr + offsets, output, mask=mask)

def silu(x):
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024 
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def test_silu():
    shape = (1024 * 10,)
    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    expected = torch.nn.functional.silu(x)
    actual = silu(x)
    assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print("✅ test_silu")

if __name__ == '__main__':
    test_silu()