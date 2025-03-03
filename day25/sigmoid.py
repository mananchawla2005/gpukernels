import triton
import triton.language as tl
import torch

@triton.jit
def sigmoid_kernel(
    input_ptr,  
    output_ptr, 
    n,          
    BLOCK_SIZE: tl.constexpr, 
):
    blockId = tl.program_id(axis=0)
    offsets = blockId * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask)
    output = 1.0 / (1.0 + tl.exp(-x))
    tl.store(output_ptr + offsets, output, mask=mask)


def sigmoid(input_tensor):
    n = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    sigmoid_kernel[grid](
        input_tensor, 
        output,
        n,
        BLOCK_SIZE,
    )
    return output