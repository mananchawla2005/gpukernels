import torch
import numpy as np
from gpu_kernels import sigmoid

def test_sigmoid():
    # Test Case 1: Simple tensor
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    output_tensor = torch.zeros_like(input_tensor)
    
    sigmoid(input_tensor, output_tensor)
    expected = torch.sigmoid(input_tensor)
    
    np.testing.assert_allclose(
        output_tensor.numpy(),
        expected.numpy(),
        rtol=1e-5,
        atol=1e-5
    )

    # Test Case 2: Larger random tensor
    input_tensor = torch.randn(1024, dtype=torch.float32)
    output_tensor = torch.zeros_like(input_tensor)
    
    sigmoid(input_tensor, output_tensor)
    expected = torch.sigmoid(input_tensor)
    
    np.testing.assert_allclose(
        output_tensor.numpy(),
        expected.numpy(),
        rtol=1e-5,
        atol=1e-5
    )
    
    print("All sigmoid tests passed!")

if __name__ == "__main__":
    test_sigmoid()