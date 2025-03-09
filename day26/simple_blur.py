import torch
import triton
import triton.language as tl
import numpy as np
import torch.nn.functional as F

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
    width32 = width.to(tl.int32)
    height32 = height.to(tl.int32)
    center_elem = stride // 2 
    y_block = tl.program_id(1) * BLOCK_SIZE_Y
    x_block = tl.program_id(0) * BLOCK_SIZE_X
    offsets_y = tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = tl.arange(0, BLOCK_SIZE_X)
    
    row = y_block + offsets_y[:, None]
    column = x_block + offsets_x[None, :]
    
    row_mask = row < height32
    column_mask = column < width32
    mask = row_mask & column_mask

    sum_r = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    sum_g = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    sum_b = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.float32)
    for i in range(-center_elem, center_elem + 1):
        y = row + i
        y_valid = (y >= 0) & (y < height32)

        for j in range(-center_elem, center_elem + 1):
            x = column + j
            x_valid = (x >= 0) & (x < width32)
            sample_mask = mask & y_valid & x_valid
            pixel_idx = (y * width32 + x) * 3
            
            # Load RGB values
            r = tl.load(Pin_ptr + pixel_idx + 0, mask=sample_mask, other=0.0)
            g = tl.load(Pin_ptr + pixel_idx + 1, mask=sample_mask, other=0.0)
            b = tl.load(Pin_ptr + pixel_idx + 2, mask=sample_mask, other=0.0)
            sum_r += r
            sum_g += g
            sum_b += b
            count += sample_mask.to(tl.float32)

    out_idx = (row * width32 + column) * 3

    r_out = (sum_r / count).to(tl.uint8)
    g_out = (sum_g / count).to(tl.uint8)
    b_out = (sum_b / count).to(tl.uint8)
    tl.store(Pout_ptr + out_idx + 0, r_out, mask=mask)
    tl.store(Pout_ptr + out_idx + 1, g_out, mask=mask)
    tl.store(Pout_ptr + out_idx + 2, b_out, mask=mask)


def simple_blur(Pin_h, Pout_h, width, height, stride):
    Pin_t = torch.from_numpy(Pin_h).clone().cuda()
    Pout_t = torch.zeros_like(Pin_t)
    
    grid_x = (width + 31) // 32
    grid_y = (height + 31) // 32
    
    simple_blur_kernel[(grid_x, grid_y)](
        Pin_t,
        Pout_t,
        width,
        height,
        stride,
    )
    
    Pout_h[:] = Pout_t.cpu().numpy()

def run_blur_test():
    height = 512
    width = 512
    stride = 3 
    Pin_h = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    Pout_h = np.empty_like(Pin_h)  # output for Triton
    
    simple_blur(Pin_h, Pout_h, width, height, stride)
    Pin_tensor = torch.from_numpy(Pin_h).permute(2, 0, 1).unsqueeze(0).cuda()
    pad = stride // 2
    kernel = torch.ones(3, 1, stride, stride, device='cuda')
    
    ones = torch.ones(1, 1, height, width, device='cuda')
    count = F.conv2d(ones, torch.ones(1, 1, stride, stride, device='cuda'), padding=pad)
    
    blurred = F.conv2d(Pin_tensor.float(), kernel, padding=pad, groups=3)
    blurred = (blurred / count).to(torch.uint8)
    
    pytorch_out = blurred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    diff = np.abs(pytorch_out.astype(np.int32) - Pout_h.astype(np.int32))
    max_diff = np.max(diff)
    print(f"Max difference between Triton and PyTorch: {max_diff}")

if __name__ == "__main__":
    run_blur_test()