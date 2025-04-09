# Day 56: AdaIN with Snake Activation

A CUDA kernel implementation that combines Instance Normalization, Adaptive Instance Normalization (AdaIN), and Snake activation function for enhanced neural network expressivity and style transfer capabilities.

## What is AdaIN Snake?

This implementation combines three powerful techniques:

1. **Instance Normalization**: Normalizes feature statistics independently for each instance in a batch
2. **Adaptive Instance Normalization (AdaIN)**: Applies style transformation by adjusting normalized features with learnable scale (gamma) and shift (beta) parameters
3. **Snake Activation**: An activation function defined as `x + (1/α) * sin²(αx)` that provides non-linearity with beneficial properties for deep networks

## Benefits

- Enables effective neural style transfer through AdaIN
- Improves gradient flow with the Snake activation function
- Maintains feature distributions through instance normalization
- Combines style control and non-linear activation in a single fused operation for better performance

## Implementation Details

The kernel processes input tensors with dimensions:
- Batch size × Channels × Width
- Normalizes features per instance
- Applies style transformation using gamma and beta parameters
- Applies Snake activation with learnable alpha parameters

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Unix/Linux
```

2. Build and install the package:

```bash
pip install .
```

## Usage Example

```python
import torch
import adain_snake_cuda

# Setup input tensors
batch_size, channels, width = 2, 16, 32
input = torch.randn(batch_size, channels, width, device='cuda')
gamma = torch.randn(channels, device='cuda')
beta = torch.randn(channels, device='cuda')
alpha = torch.ones(channels, device='cuda')

# Calculate mean and variance
mean = input.mean(dim=2)
var = input.var(dim=2)

# Apply AdaIN Snake
output = adain_snake_cuda.forward(input, gamma, beta, mean, var, alpha)
```

## References

- [AdaIN paper: Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- [Snake: A Novel Activation Function](https://arxiv.org/abs/2006.08195)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)