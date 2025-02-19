import torch
import gpu_kernels
import numpy as np

def test_gelu():
    # Test parameters
    batch_size = 1024
    torch.manual_seed(42)
    
    # Generate random input tensor
    input_tensor = torch.randn(batch_size, requires_grad=True)
    custom_input = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass - PyTorch
    pytorch_output = torch.nn.functional.gelu(input_tensor)
    
    # Forward pass - Custom CUDA kernel
    custom_output = torch.empty_like(input_tensor)
    gpu_kernels.gelu_forward(custom_input, custom_output)
    
    # Check forward pass accuracy
    max_diff = torch.max(torch.abs(pytorch_output - custom_output))
    print(f"Forward pass maximum difference: {max_diff:.6f}")
    assert max_diff < 1e-2, "Forward pass results don't match!"
    # Backward pass
    # Create random gradient for backward pass
    grad_output = torch.randn_like(pytorch_output)
    
    # PyTorch backward
    pytorch_output.backward(grad_output)
    pytorch_grad = input_tensor.grad
    
    # Custom CUDA kernel backward
    custom_grad = torch.empty_like(custom_input)
    gpu_kernels.gelu_backward(grad_output, custom_input, custom_grad)
    
    # Check backward pass accuracy
    max_grad_diff = torch.max(torch.abs(pytorch_grad - custom_grad))
    print(f"Backward pass maximum difference: {max_grad_diff:.6f}")
    assert max_grad_diff < 1e-2, "Backward pass results don't match!"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    try:
        test_gelu()
    except AssertionError as e:
        print(f"Test failed: {e}")