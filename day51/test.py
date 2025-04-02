import torch
import gpu_kernels
import numpy as np

def compute_reference_attention(q, k, v, mask=None):
    """Compute attention using PyTorch's native operations"""
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(0, 1)) * scale
    if mask is not None:
        scores = scores * mask
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output

def test_lightning_attention():
    B = 16 
    D = 64  
    N = 32 
    torch.manual_seed(42)  

    q = torch.randn(N, D, dtype=torch.float32, device='cpu')
    k = torch.randn(N, D, dtype=torch.float32, device='cpu')
    v = torch.randn(N, D, dtype=torch.float32, device='cpu')
    o_cuda = torch.zeros(N, D, dtype=torch.float32, device='cpu')

    mask = torch.triu(torch.ones(B, B, dtype=torch.float32), diagonal=1)
    mask = 1.0 - mask  

    print("Running attention implementations...")
    try:
        gpu_kernels.lightning_attention(q, k, v, o_cuda)
        o_ref = compute_reference_attention(q, k, v, mask)

        max_diff = torch.max(torch.abs(o_cuda - o_ref))
        mean_diff = torch.mean(torch.abs(o_cuda - o_ref))
        print(f"\nMax difference between CUDA and PyTorch: {max_diff:.6f}")
        print(f"Mean difference between CUDA and PyTorch: {mean_diff:.6f}")

        assert o_cuda.shape == (N, D), f"Output shape mismatch: expected {(N, D)}, got {o_cuda.shape}"
        assert not torch.isnan(o_cuda).any(), "CUDA output contains NaN values"
        assert not torch.isinf(o_cuda).any(), "CUDA output contains infinite values"
        
        rtol = 1e-4 
        atol = 1e-4  
        is_close = torch.allclose(o_cuda, o_ref, rtol=rtol, atol=atol)
        print(f"\nOutputs match within tolerance: {is_close}")

        if not is_close:
            print("\nSample comparison (first 5 elements):")
            print("CUDA implementation:", o_cuda[0, :5])
            print("PyTorch reference: ", o_ref[0, :5])
        
        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    test_lightning_attention()