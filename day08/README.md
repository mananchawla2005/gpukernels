# Day 8:
# TASK: Batch Normalization Implementation

A CUDA kernel implementing batch normalization, a technique to normalize layer inputs and accelerate training in deep neural networks.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY BATCH NORMALIZATION?

Batch normalization reduces internal covariate shift by normalizing each spatial location (H, W) across the batch dimension. Its benefits include:
- Faster convergence during training
- Improved training stability
- Ability to use higher learning rates
- Reduced sensitivity to network initialization

The normalization is performed as:
  Y = gamma * (X - mean) / sqrt(variance + epsilon) + beta

Where mean and variance are computed for each (H, W) position across all samples in the batch.

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

## Testing and Usage Example
Run the test script to verify the installation:

```bash
python test.py
```

- You can also benchmark the implementation using the provided benchmark script.

**Note**: Ensure you're in the activated virtual environment when running the code.

# External References

[Batch Normalization Paper](https://arxiv.org/abs/1502.03167)  
[PyTorch BatchNorm Documentation](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
