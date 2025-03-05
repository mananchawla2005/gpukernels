import triton
import triton.language as tl
import torch
import math
@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    
    pid = tl.program_id(axis=0)
    
    
    block_start = pid * BLOCK_SIZE
    
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    
    mask = offsets < n_elements
    
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    
    sqrt_2 = tl.sqrt(tl.float32(2.0))
    result = 0.5 * x * (1.0 + tl.erf(x / sqrt_2))
    
    
    tl.store(output_ptr + offsets, result, mask=mask)
def gelu(x):
    """
    Applies GELU activation function to the input tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Output tensor with GELU activation applied
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    
    BLOCK_SIZE = 1024
    
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    
    gelu_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    
    return output
if __name__ == "__main__":
    
    x = torch.linspace(-3, 3, 10000, device='cuda')
    
    
    triton_output = gelu(x)
    
    
    torch_output = torch.nn.functional.gelu(x)
    
    
    diff = torch.abs(triton_output - torch_output).max().item()
    print(f"Maximum difference between Triton and PyTorch GELU: {diff}")