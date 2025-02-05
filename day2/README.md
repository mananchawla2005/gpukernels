# Day 2: 
# TASK 1: CUDA Vector Addition with Python Bindings

A CUDA vector addition kernel wrapped with PyTorch C++ extensions for use in Python.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## Setup
1. Create and activate a virtual environment:

```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Unix/Linux
```

2. Install the package in the environment

```bash
pip install .
```

## Testing
Run the test script to verify the installation:

```python
python test.py
```

The test script will:
1. Create two tensors filled with ones
2. Perform vector addition using the CUDA kernel
3. Print the resulting tensor (should show all 2's)

## Usage Example
```python
import torch
from gpu_kernels import vec_add

# Create input tensors
A = torch.ones(1000)    # First vector
B = torch.ones(1000)    # Second vector
C = torch.zeros_like(A) # Output vector

# Perform vector addition on GPU
vec_add(A, B, C)

# Print result
print(C)
```

**Note**: Make sure you're in the activated virtual environment when running the code.