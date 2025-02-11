import torch
import numpy as np
from gpu_kernels import batch_norm

def test_batch_norm():
    epsilon = 1e-5

    # Test Case 1: Simple tensor
    input_tensor = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ], dtype=torch.float32).cuda()
    
    output_tensor = torch.zeros_like(input_tensor)
    gamma = torch.ones((2, 2), dtype=torch.float32).cuda()
    beta = torch.zeros((2, 2), dtype=torch.float32).cuda()
    
    batch_norm(input_tensor, output_tensor, gamma, beta, epsilon)
    
    mean = input_tensor.mean(dim=0)  # shape (H,W)
    var = input_tensor.var(dim=0, unbiased=False)
    expected = gamma * (input_tensor - mean) / torch.sqrt(var + epsilon) + beta
    
    np.testing.assert_allclose(
        output_tensor.cpu().numpy(),
        expected.cpu().numpy(),
        rtol=1e-5,
        atol=1e-5
    )
    
    input_tensor = torch.randn(32, 16, 16, dtype=torch.float32).cuda()
    output_tensor = torch.zeros_like(input_tensor)
    gamma = torch.ones((16, 16), dtype=torch.float32).cuda()
    beta = torch.zeros((16, 16), dtype=torch.float32).cuda()
    
    batch_norm(input_tensor, output_tensor, gamma, beta, epsilon)
    
    mean = input_tensor.mean(dim=0)
    var = input_tensor.var(dim=0, unbiased=False)
    expected = gamma * (input_tensor - mean) / torch.sqrt(var + epsilon) + beta
    
    np.testing.assert_allclose(
        output_tensor.cpu().numpy(),
        expected.cpu().numpy(),
        rtol=1e-5,
        atol=1e-5
    )
    
    print("All batch norm tests passed!")

if __name__ == "__main__":
    test_batch_norm()