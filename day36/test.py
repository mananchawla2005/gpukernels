import torch
import numpy as np
from gpu_kernels import dynamic_tanh

def test_dynamic_tanh():
    # Test Case 1: Simple tensor with alpha=1.0 and bias=0
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    weight_tensor = torch.ones_like(input_tensor)
    output_tensor = torch.zeros_like(input_tensor)
    bias_tensor = torch.zeros_like(input_tensor)
    alpha = 1.0
    
    dynamic_tanh(input_tensor, weight_tensor, output_tensor, alpha, bias_tensor)
    expected = torch.tanh(alpha * input_tensor) * weight_tensor + bias_tensor
    
    np.testing.assert_allclose(
        output_tensor.numpy(),
        expected.numpy(),
        rtol=1e-5,
        atol=1e-5
    )
    print("Test Case 1 (basic test) passed!")

    # Test Case 2: With custom alpha, weights and bias
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    weight_tensor = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float32)
    output_tensor = torch.zeros_like(input_tensor)
    bias_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    alpha = 2.0
    
    dynamic_tanh(input_tensor, weight_tensor, output_tensor, alpha, bias_tensor)
    expected = torch.tanh(alpha * input_tensor) * weight_tensor + bias_tensor
    
    np.testing.assert_allclose(
        output_tensor.numpy(),
        expected.numpy(),
        rtol=1e-5,
        atol=1e-5
    )
    print("Test Case 2 (custom parameters) passed!")

    # Test Case 3: Larger random tensor
    size = 1024
    input_tensor = torch.randn(size, dtype=torch.float32)
    weight_tensor = torch.randn(size, dtype=torch.float32)
    output_tensor = torch.zeros_like(input_tensor)
    bias_tensor = torch.randn(size, dtype=torch.float32)
    alpha = 1.5
    
    dynamic_tanh(input_tensor, weight_tensor, output_tensor, alpha, bias_tensor)
    expected = torch.tanh(alpha * input_tensor) * weight_tensor + bias_tensor
    
    np.testing.assert_allclose(
        output_tensor.numpy(),
        expected.numpy(),
        rtol=1e-5,
        atol=1e-5
    )
    print("Test Case 3 (large random tensor) passed!")

    # # Test Case 4: Edge cases with very large and small values
    # input_tensor = torch.tensor([1e-6, 1e-3, 1.0, 1e3, 1e6], dtype=torch.float32)
    # weight_tensor = torch.ones_like(input_tensor)
    # output_tensor = torch.zeros_like(input_tensor)
    # bias_tensor = torch.zeros_like(input_tensor)
    # alpha = 0.5
    
    # dynamic_tanh(input_tensor, weight_tensor, output_tensor, alpha, bias_tensor)
    # expected = torch.tanh(alpha * input_tensor) * weight_tensor + bias_tensor
    
    # np.testing.assert_allclose(
    #     output_tensor.numpy(),
    #     expected.numpy(),
    #     rtol=1e-5,
    #     atol=1e-5
    # )
    # print("Test Case 4 (edge cases) passed!")
    
    print("All dynamic tanh tests passed!")

if __name__ == "__main__":
    test_dynamic_tanh()