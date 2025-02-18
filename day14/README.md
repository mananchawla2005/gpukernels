# Day 14:
# TASK: Softmax CUDA Implementation

This task implements an efficient softmax operation using CUDA. Softmax is a crucial activation function in deep learning, particularly in attention mechanisms and classification tasks.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY SOFTMAX?

Softmax is essential for many reasons:
- Converts raw scores into probability distributions
- Key component in attention mechanisms in transformers
- Used in classification tasks for output normalization
- Critical for stable numerical computations in deep learning
- Requires careful implementation to avoid numerical overflow

## Implementation Features
- Efficient parallel reduction for max finding
- Shared memory usage for intermediate results
- Numerically stable computation
- Handles arbitrary batch and sequence lengths
- Block-level synchronization for accurate results

## Performance Results
Our implementation shows competitive performance:
- PyTorch GPU: ~0.124 ms
- Custom CUDA: ~5.172 ms (including memory transfers)
  - Kernel execution: 0.464 ms
  - Memory transfers: ~1.541 ms

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
Run the benchmark script to compare implementations:

```bash
python day14_benchmark.py
```

The benchmark will:
- Compare PyTorch CPU, GPU, and custom CUDA implementations
- Show timing information for each implementation
- Verify numerical accuracy between implementations

Make sure you are in the activated virtual environment when running the code.

## Implementation Details
Our CUDA kernel includes:
- Block-level max finding using shared memory
- Parallel exp computation
- Reduction for sum calculation
- Final normalization step

## External References

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
[Softmax in Deep Learning](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
