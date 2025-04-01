import torch
import gpu_kernels
import math

def pytorch_self_attention(q, k, v, d):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

def test_attention_backward():
    batch_size = 32
    d = 64  
    torch.manual_seed(42)  
    query = torch.randn(batch_size, d, device='cuda')
    key = query.clone()  # 
    value = query.clone()
    
    query_pt = query.clone().detach().requires_grad_(True)
    key_pt = query_pt.clone().detach().requires_grad_(True)  
    value_pt = query_pt.clone().detach().requires_grad_(True)  
    
    output_pt, attn_weights = pytorch_self_attention(query_pt, key_pt, value_pt, d)
    
    grad_output = torch.randn_like(output_pt)
    
    output_pt.backward(grad_output)
    
    grad_q_cuda = torch.zeros_like(query)
    grad_k_cuda = torch.zeros_like(query)
    grad_v_cuda = torch.zeros_like(query)
    
    gpu_kernels.self_attention_backward(
        grad_output, query, key, value,
        attn_weights,
        grad_q_cuda, grad_k_cuda, grad_v_cuda,
        d
    )
    
    rtol, atol = 1e-4, 1e-4
    print(f"Query grad max diff: {(query_pt.grad - grad_q_cuda).abs().max().item()}")
    print(f"Key grad max diff: {(key_pt.grad - grad_k_cuda).abs().max().item()}")
    print(f"Value grad max diff: {(value_pt.grad - grad_v_cuda).abs().max().item()}")
    
    assert torch.allclose(query_pt.grad, grad_q_cuda, rtol=rtol, atol=atol), \
        f"Query gradients don't match! Max diff: {(query_pt.grad - grad_q_cuda).abs().max()}"
    assert torch.allclose(key_pt.grad, grad_k_cuda, rtol=rtol, atol=atol), \
        f"Key gradients don't match! Max diff: {(key_pt.grad - grad_k_cuda).abs().max()}"
    assert torch.allclose(value_pt.grad, grad_v_cuda, rtol=rtol, atol=atol), \
        f"Value gradients don't match! Max diff: {(value_pt.grad - grad_v_cuda).abs().max()}"
    
    print("All gradients match within tolerance!")

if __name__ == "__main__":
    test_attention_backward()