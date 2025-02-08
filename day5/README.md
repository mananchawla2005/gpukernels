# Day 5:
# TASK: SIMPLE MATRIX MULTIPLICATION
A CUDA kernel to perform simple matrix multiplication between two matrices.

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

2. Build and install the package:

```bash
pip install .
```

## Testing and Usage Example
Run the test scripts to verify the installation:

```python
python test.py
```

- Additionally we can benchmark the implementation using the [benchmark script](../experiments/day5_benchmark.py)

**Note**: Ensure you're in the activated virtual environment when running the code.