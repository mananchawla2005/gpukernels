import torch
from gpu_kernels import rgb_to_grayscale
import numpy as np

def test_rgb_to_grayscale():
    input_data = torch.zeros((3, 3, 3), dtype=torch.uint8)
    input_data[0,0] = torch.tensor([255, 0, 0], dtype=torch.uint8)
    input_data[1,1] = torch.tensor([0, 255, 0], dtype=torch.uint8)
    input_data[2,2] = torch.tensor([0, 0, 255], dtype=torch.uint8)
    
    output = torch.zeros((3, 3), dtype=torch.uint8)
    
    rgb_to_grayscale(input_data, output)
    
    expected = np.array([
        [0.21*255, 0, 0],  
        [0, 0.72*255, 0], 
        [0, 0, 0.07*255]   
    ], dtype=np.uint8)
    
    np.testing.assert_array_almost_equal(output.numpy(), expected)

if __name__ == "__main__":
    test_rgb_to_grayscale()
    print("All tests passed!")