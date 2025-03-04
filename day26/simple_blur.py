import torch
import triton
import triton.language as tl

@triton.jit
def simple_blur_kernel(
    Pin_ptr,        
    Pout_ptr,      
    width,     
    height,           
    stride,            
    BLOCK_SIZE_X: tl.constexpr = 32,
    BLOCK_SIZE_Y: tl.constexpr = 32,
):
    row = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    column = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    
    row_mask = row < height
    column_mask = column < width
    
    center_elem = stride // 2
    
    sum_r = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    sum_g = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    sum_b = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    
    for i in range(-center_elem, center_elem + 1):
        y = row + i
        y_valid = (y >= 0) & (y < height)
        
        for j in range(-center_elem, center_elem + 1):
            x = column + j
            x_valid = (x >= 0) & (x < width)
            
            mask = row_mask & column_mask & y_valid & x_valid
            
            pixel_idx = (y * width + x) * 3
            
            r = tl.load(Pin_ptr + pixel_idx, mask=mask, other=0.0)
            g = tl.load(Pin_ptr + pixel_idx + 1, mask=mask, other=0.0)
            b = tl.load(Pin_ptr + pixel_idx + 2, mask=mask, other=0.0)
            
            sum_r += r
            sum_g += g
            sum_b += b
            count += mask.to(tl.float32)
    
    out_idx = (row * width + column) * 3
    
    mask = row_mask & column_mask
    tl.store(Pout_ptr + out_idx, (sum_r / count).to(tl.uint8), mask=mask)
    tl.store(Pout_ptr + out_idx + 1, (sum_g / count).to(tl.uint8), mask=mask)
    tl.store(Pout_ptr + out_idx + 2, (sum_b / count).to(tl.uint8), mask=mask)


def simple_blur(Pin_h, Pout_h, width, height, stride):
    Pin_t = torch.from_numpy(Pin_h).clone()
    Pout_t = torch.zeros_like(Pin_t)
    
    grid_x = (width + 31) // 32
    grid_y = (height + 31) // 32
    
    simple_blur_kernel[(grid_x, grid_y)](
        Pin_t.data_ptr(),
        Pout_t.data_ptr(),
        width,
        height,
        stride,
    )
    
    Pout_h[:] = Pout_t.numpy()