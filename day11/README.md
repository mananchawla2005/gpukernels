# Day 11:
# TASK: Dynamic Tiled Matrix Multiplication Implementation

This task implements matrix multiplication using dynamically-sized tiles in CUDA. The tile size is automatically optimized based on the GPU's specifications to maximize performance.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY DYNAMIC TILING?

Dynamic tiled matrix multiplication:
- Automatically determines optimal tile size based on:
  - Available shared memory
  - Maximum threads per block
  - Matrix dimensions
- Adapts to different GPU architectures
- Optimizes memory usage and occupancy
- Improves performance portability across devices

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

[CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
[CUDA Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)