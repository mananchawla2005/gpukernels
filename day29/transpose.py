import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    width, height,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    offs_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)  # shape: (BLOCK_SIZE_X,)
    offs_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)  # shape: (BLOCK_SIZE_Y,)
    
    x = offs_x[None, :]  # shape: (1, BLOCK_SIZE_X)
    y = offs_y[:, None]  # shape: (BLOCK_SIZE_Y, 1)
    mask = (x < width) & (y < height)
    input_indices = y * width + x  # Broadcasting to shape (BLOCK_SIZE_Y, BLOCK_SIZE_X)
    a = tl.load(input_ptr + input_indices, mask=mask, other=0.)
    output_indices = x * height + y
    tl.store(output_ptr + output_indices, a, mask=mask)

def transpose(input_h, output_h, width, height):
    input_t = torch.tensor(input_h, dtype=torch.float32, device='cuda')
    output_t = torch.empty((width * height,), dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32
    grid = (triton.cdiv(width, BLOCK_SIZE_X), triton.cdiv(height, BLOCK_SIZE_Y))
    
    transpose_kernel[grid](
        input_t, output_t, 
        width, height, 
        BLOCK_SIZE_X=BLOCK_SIZE_X, BLOCK_SIZE_Y=BLOCK_SIZE_Y
    )
    
    output_h[:] = output_t.cpu().numpy().reshape(width, height)

def demo():
    # Create a simple input matrix of shape (height, width)
    height, width = 3, 4
    input_np = np.array([[float(i + j * width) for i in range(width)] for j in range(height)], dtype=np.float32)
    # Allocate output container with transposed shape (width, height)
    output_np = np.empty((width, height), dtype=np.float32)
    
    print("Input Matrix:")
    print(input_np)
    
    # Call the transpose function; note that torch.tensor accepts numpy arrays.
    transpose(input_np, output_np, width, height)
    
    print("\nTransposed Matrix:")
    print(output_np)

if __name__ == '__main__':
    demo()