# Day 12:
# TASK: Layer Normalization CUDA Implementation

This task implements an efficient Layer Normalization operation using CUDA. Layer normalization is a crucial technique in deep learning that normalizes the inputs across the features.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY LAYER NORMALIZATION?

Layer normalization offers several benefits:
- Stabilizes deep neural network training
- Reduces internal covariate shift
- Enables faster training convergence
- Works well with recurrent neural networks
- Independent of batch size

## Implementation Features
- Shared memory utilization for faster access
- Per-row normalization across features
- Efficient parallel reduction for mean and variance
- Numerically stable variance computation
- Epsilon factor for numerical stability

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

Make sure you are in the activated virtual environment when running the code.

# External References

[Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
[CUDA Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)