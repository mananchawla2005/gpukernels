import torch
import numpy as np
from gpu_kernels import gelu

def test_gelu():
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], 
                              dtype=torch.float32).reshape(1, -1).cuda()
    output_tensor = torch.zeros_like(input_tensor)
    
    gelu(input_tensor, output_tensor)
    
    expected = torch.nn.functional.gelu(input_tensor)
    
    np.testing.assert_allclose(
        output_tensor.cpu().numpy(), 
        expected.cpu().numpy(), 
        rtol=1e-5, 
        atol=1e-5
    )
    
    input_tensor = torch.randn(1, 1000, dtype=torch.float32).cuda()
    output_tensor = torch.zeros_like(input_tensor)
    
    gelu(input_tensor, output_tensor)
    expected = torch.nn.functional.gelu(input_tensor)
    
    np.testing.assert_allclose(
        output_tensor.cpu().numpy(), 
        expected.cpu().numpy(), 
        rtol=1e-5, 
        atol=1e-5
    )
    
    print("All tests passed!")

if __name__ == "__main__":
    test_gelu()