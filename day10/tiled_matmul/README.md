# Day 10:
# TASK 2: Tiled Matrix Multiplication Implementation

This task implements tiled matrix multiplication using a CUDA kernel. The tiled approach divides matrices into smaller sub-matrices (tiles) to improve memory access patterns and performance on NVIDIA GPUs.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY TILED MATRIX MULTIPLICATION?

Tiled matrix multiplication:
- Breaks a large matrix multiplication problem into smaller sub-problems (tiles)
- Improves memory coalescing by loading tiles into shared memory
- Reduces global memory accesses
- Scales better with larger matrix sizes

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
python test_matmul.py
```

Make sure you are in the activated virtual environment when running the code.

# External References

[Tiled Matrix Multiplication Article](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-matrix-multiplication/)
[PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
