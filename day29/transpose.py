import torch
import triton
import triton.language as tl
@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    width, height,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    start_x = pid_x * BLOCK_SIZE_X
    start_y = pid_y * BLOCK_SIZE_Y
    
    offs_x = start_x + tl.arange(0, BLOCK_SIZE_X)
    offs_y = start_y + tl.arange(0, BLOCK_SIZE_Y)
    
    mask_x = offs_x < width
    mask_y = offs_y < height
    
    for i in range(BLOCK_SIZE_Y):
        y = start_y + i
        if y < height:
            for j in range(BLOCK_SIZE_X):
                x = start_x + j
                if x < width:
                    input_idx = y * width + x
                    output_idx = x * height + y
                    
                    val = tl.load(input_ptr + input_idx)
                    tl.store(output_ptr + output_idx, val)

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
    
    output_h[:] = output_t.cpu().numpy()