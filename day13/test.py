import torch
import gpu_kernels

# Create input tensor
input_tensor = torch.randn(100, 200, dtype=torch.float32)
# Create output tensor with transposed shape
output_tensor = torch.empty(200, 100, dtype=torch.float32)
# Call the CUDA transpose
gpu_kernels.transpose(input_tensor, output_tensor)
print(input_tensor.transpose(0, 1))
print(output_tensor)