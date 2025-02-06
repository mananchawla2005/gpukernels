import torch
from gpu_kernels import rgb_to_grayscale
import numpy as np
from PIL import Image

img = Image.open('dog.jpeg').convert('RGB')
rgb_array = np.array(img)

input_tensor = torch.from_numpy(rgb_array).byte()
input_tensor = input_tensor.reshape(rgb_array.shape[0], rgb_array.shape[1], 3)
output_tensor = torch.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=torch.uint8)

# Swap the order of arguments: the input image tensor comes first.
rgb_to_grayscale(input_tensor, output_tensor)

result = Image.fromarray(output_tensor.numpy())
result.save('dog_grayscale.jpeg')