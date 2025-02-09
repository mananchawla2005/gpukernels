# Day 5:
# TASK: Global Memory Coalescing MATRIX MULTIPLICATION
A CUDA kernel to perform coalased matrix multiplication between two matrices.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY IT IS BETTER?

Thread Layout in 2D block:
(0,0) (0,1) ... (0,15)
(1,0) (1,1) ... (1,15)
...
(15,0) (15,1) ... (15,15)


Row 0: threads (0,0)-(0,15) → addresses 0,1,2,...,15
Row 1: threads (1,0)-(1,15) → addresses N,N+1,N+2,...,N+15


This means consecutive threads in a warp are accessing elements that are N elements apart in memory
Not coalesced because memory accesses aren't contiguous within a warp

- Why 1D is Better
In 1D version, consecutive thread IDs map to consecutive memory locations
Thread 0-31 access consecutive memory locations
This allows the GPU to coalesce these accesses into fewer memory transactions


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
Run the test scripts to verify the installation:

```python
python test.py
```

- Additionally we can benchmark the implementation using the [benchmark script](../experiments/day6_benchmark.py)

**Note**: Ensure you're in the activated virtual environment when running the code.