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
    output = 1.0 / (1.0 + tl.exp(-x)) # storing in register
    tl.store(output_ptr + offsets, output, mask=mask) # storing in dram


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

def test_sigmoid():
    """Test sigmoid implementation with a random tensor"""
    # Create a random input tensor
    size = 1000000  # 1M elements
    x = torch.randn(size, device='cuda')
    
    # Time the PyTorch implementation
    torch_start = torch.cuda.Event(enable_timing=True)
    torch_end = torch.cuda.Event(enable_timing=True)
    torch_start.record()
    torch_output = torch.sigmoid(x)
    torch_end.record()
    torch.cuda.synchronize()
    torch_time = torch_start.elapsed_time(torch_end)
    
    # Time the Triton implementation
    triton_start = torch.cuda.Event(enable_timing=True)
    triton_end = torch.cuda.Event(enable_timing=True)
    triton_start.record()
    triton_output = sigmoid(x)
    triton_end.record()
    torch.cuda.synchronize()
    triton_time = triton_start.elapsed_time(triton_end)
    
    # Verify correctness
    max_diff = torch.max(torch.abs(torch_output - triton_output))
    
    print(f"Input shape: {x.shape}")
    print(f"PyTorch sigmoid time: {torch_time:.4f}ms")
    print(f"Triton sigmoid time: {triton_time:.4f}ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    print(f"Maximum difference: {max_diff:.6f}")
    print("\nFirst 5 elements:")
    print(f"Input: {x[:5]}")
    print(f"PyTorch output: {torch_output[:5]}")
    print(f"Triton output: {triton_output[:5]}")

if __name__ == "__main__":
    test_sigmoid()