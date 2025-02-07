import torch
from gpu_kernels import simple_blur
import numpy as np
from PIL import Image

img = Image.open('../../day3/dog.jpeg').convert('RGB')
rgb_array = np.array(img)

input_tensor = torch.from_numpy(rgb_array).byte()
input_tensor = input_tensor.reshape(rgb_array.shape[0], rgb_array.shape[1], 3)
output_tensor = torch.zeros_like(input_tensor, dtype=torch.uint8)

simple_blur(input_tensor, output_tensor, 50)

result = Image.fromarray(output_tensor.numpy())
result.save('dog_blurred.jpeg')