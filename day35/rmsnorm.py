import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    hidden_dim,
    eps,
    stride_batch,
    stride_seq,
    stride_hidden,
    weight_stride,
    num_batches,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= num_batches:
        return
    
    batch_seq_offset = batch_idx * stride_batch + seq_idx * stride_seq
    x_ptr = x_ptr + batch_seq_offset
    output_ptr = output_ptr + batch_seq_offset
    
    sum_squares = 0.0
    
    for i in range(0, hidden_dim, BLOCK_SIZE):
        block_offset = i
        
        mask = block_offset + tl.arange(0, BLOCK_SIZE) < hidden_dim
        x_block = tl.load(x_ptr + block_offset * stride_hidden, mask=mask, other=0.0)
        
        sum_squares += tl.sum(x_block * x_block, axis=0)
        
    rms = tl.sqrt(sum_squares / hidden_dim + eps)
    inv_rms = 1.0 / rms
    
    for i in range(0, hidden_dim, BLOCK_SIZE):
        block_offset = i
        
        mask = block_offset + tl.arange(0, BLOCK_SIZE) < hidden_dim
        x_block = tl.load(x_ptr + block_offset * stride_hidden, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + block_offset * weight_stride, mask=mask, other=0.0)
        
        result_block = x_block * inv_rms * weight_block
        tl.store(output_ptr + block_offset * stride_hidden, result_block, mask=mask)

def rmsnorm(x, weight, eps=1e-6):
    batch_size, seq_len, hidden_dim = x.shape
    output = torch.empty_like(x)
    
    stride_batch = x.stride(0)
    stride_seq = x.stride(1)
    stride_hidden = x.stride(2)
    weight_stride = weight.stride(0)
    
    grid = (batch_size * seq_len,)
    
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    rmsnorm_kernel[grid](
        x, weight, output,
        hidden_dim, eps,
        stride_batch, stride_seq, stride_hidden,
        weight_stride,
        batch_size, seq_len,
        BLOCK_SIZE,
    )
    
    return output

if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    hidden_dim = 768
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    weight = torch.ones(hidden_dim, device='cuda')
    
    custom_output = rmsnorm(x, weight)
    
    def rmsnorm_ref(x, weight, eps=1e-6):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms * weight
    
    ref_output = rmsnorm_ref(x, weight)
    
    max_diff = torch.max(torch.abs(custom_output - ref_output))
    print(f"Maximum difference between custom and reference implementation: {max_diff:.6f}")
    assert max_diff < 1e-5, "Implementation doesn't match reference!"
    print("Test passed!")