import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def layer_norm_kernel(
    input_ptr, output_ptr,
    stride, n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride
    out_row_start_ptr = output_ptr + row_idx * stride
    mean = 0.0
    var = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        col_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    
    mean = mean / n_cols
    for off in range(0, n_cols, BLOCK_SIZE):
        col_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        var += tl.sum((x - mean) * (x - mean), axis=0)
    var = var / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    for off in range(0, n_cols, BLOCK_SIZE):
        col_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start_ptr + col_offsets, mask=mask)
        
        y = (x - mean) * rstd
        tl.store(out_row_start_ptr + col_offsets, y, mask=mask)
def layer_norm(input_h, output_h, rows, cols):
    input_t = torch.tensor(input_h, dtype=torch.float32, device='cuda')
    output_t = torch.empty_like(input_t)
    grid = (rows,)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(cols))
    layer_norm_kernel[grid](
        input_t, output_t,
        input_t.stride(0), cols,  
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    output_h[:] = output_t.cpu().numpy()

def demo():
    rows, cols = 4, 8
    np.random.seed(42)
    input_np = np.random.randn(rows, cols).astype(np.float32)
    output_np = np.empty_like(input_np)
    
    print("Input Matrix:")
    print(input_np)
    layer_norm(input_np, output_np, rows, cols)
    
    print("\nOutput Matrix (Triton Layer Normalization):")
    print(output_np)
    input_torch = torch.tensor(input_np, device='cuda')
    torch_ln = torch.nn.LayerNorm(cols, eps=1e-5).to('cuda')
    output_torch = torch_ln(input_torch).detach().cpu().numpy()
    
    print("\nOutput Matrix (PyTorch Layer Normalization):")
    print(output_torch)
    diff = np.abs(output_np - output_torch).max()
    print(f"\nMaximum absolute difference: {diff:.8f}")
    is_close = np.allclose(output_np, output_torch, rtol=1e-5, atol=1e-5)
    print(f"Results are {'close' if is_close else 'different'}")

if __name__ == '__main__':
    demo()