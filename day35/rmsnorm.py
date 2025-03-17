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
    stride_batch_seq, 
    stride_hidden,
    num_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    if pid >= num_elements:
        return
    offset = pid * stride_batch_seq
    
    sum_squares = 0.0
    
    for i in range(0, hidden_dim, BLOCK_SIZE):
        block_idx = tl.arange(0, BLOCK_SIZE)
        mask = (i + block_idx) < hidden_dim
        
        x = tl.load(x_ptr + offset + (i + block_idx) * stride_hidden, mask=mask, other=0.0)
        sum_squares += tl.sum(x * x, axis=0)
    
    # Calculate RMS normalization factor: 1/sqrt((1/N)*sum(x^2) + eps)
    inv_rms = 1.0 / tl.sqrt(sum_squares / hidden_dim + eps)
    
    for i in range(0, hidden_dim, BLOCK_SIZE):
        block_idx = tl.arange(0, BLOCK_SIZE)
        mask = (i + block_idx) < hidden_dim
        
        x = tl.load(x_ptr + offset + (i + block_idx) * stride_hidden, mask=mask, other=0.0)
        gamma = tl.load(weight_ptr + (i + block_idx), mask=mask, other=0.0)
        
        y = x * inv_rms * gamma
        
        tl.store(output_ptr + offset + (i + block_idx) * stride_hidden, y, mask=mask)

def rmsnorm(x, weight, eps=1e-6):
    if x.dim() == 3:
        batch_size, seq_len, hidden_dim = x.shape
        x_reshaped = x.view(-1, hidden_dim) 
    else:
        x_reshaped = x
        hidden_dim = x.shape[-1]
    
    num_elements = x_reshaped.shape[0]
    output = torch.empty_like(x)
    
    stride_batch_seq = x_reshaped.stride(0)
    stride_hidden = x_reshaped.stride(1)
    grid = (num_elements,)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)
    
    rmsnorm_kernel[grid](
        x_reshaped, weight, output.view(-1, hidden_dim),
        hidden_dim, eps,
        stride_batch_seq, stride_hidden,
        num_elements,
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
        # RMSNorm(x_i) = (x_i / sqrt((1/N) * Σ(j=1 to N) x_j^2 + ε)) * γ
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms * weight
    
    ref_output = rmsnorm_ref(x, weight)
    
    max_diff = torch.max(torch.abs(custom_output - ref_output))
    print(f"Maximum difference between custom and reference implementation: {max_diff:.6f}")
    assert max_diff < 1e-5, "Implementation doesn't match reference!"
    print("Test passed!")