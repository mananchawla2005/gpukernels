import torch
import gpu_kernels
import time
import matplotlib.pyplot as plt

# Image dimensions
width = 512
height = 512

# Create an output tensor
output = torch.empty((height, width, 4), dtype=torch.uint8)

# Run the kernel
start_time = time.time()
gpu_kernels.generate_image(output, width, height, 0.0)
end_time = time.time()

print(f"Kernel execution time: {end_time - start_time:.4f} seconds")

# Display the image
plt.imshow(output.numpy())
plt.axis('off')  # Hide axes
plt.show()

# Animate the image
num_frames = 100
animation_delay = 0.05  # Delay between frames in seconds

for i in range(num_frames):
    current_time = i * 0.1
    gpu_kernels.generate_image(output, width, height, current_time)
    plt.imshow(output.numpy())
    plt.title(f"Time: {current_time:.2f}")
    plt.pause(animation_delay)

plt.close()  # Close the plot after the animation