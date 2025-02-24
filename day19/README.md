# Day 19: Custom CUDA Self-Attention Implementation

This project implements an efficient CUDA-based self-attention mechanism similar to the one used in transformer architectures. The implementation features tiled matrix multiplication and optimized softmax computation.

## What is Self-Attention?
Self-attention is a key component in transformer architectures that:
- Computes attention scores between all pairs of elements in a sequence
- Applies softmax normalization to scores
- Uses scores to create weighted combinations of values
- Enables modeling of long-range dependencies

## Implementation Features
- Tiled matrix multiplication for Q×K^T and attention×V
- Parallel softmax computation with:
  - Block-wise maximum finding
  - Numerically stable exponential
  - Efficient parallel reduction for sum
- PyTorch integration via pybind11

## Key Components
1. **CUDA Kernels**:
   - `tiled_matmul_kernel`: Efficient matrix multiplication using shared memory
   - `softmax_kernel`: Parallel softmax computation
   - Support for transposed matrix operations
   - Memory-efficient implementation

2. **PyTorch Binding**:
   - Seamless integration with PyTorch tensors
   - Proper CUDA memory management
   - Type checking and validation

3. **Python Interface**:
   - Simple API for self-attention computation
   - Compatible with PyTorch tensors
   - Easy comparison with PyTorch's native implementation

## Performance Notes
- Uses tiled approach for better memory locality
- Implements numerically stable softmax
- Handles arbitrary input dimensions
- Supports scale factor (1/√d) for attention scores

## Prerequisites
- CUDA-capable GPU
- CUDA Toolkit 11.0+
- Python 3.8+
- PyTorch 2.0+

## Installation
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch

# Build and install the package
pip install .
```

## Usage Example
```python
import torch
import gpu_kernels

# Create input tensor
x = torch.randn(32, 64, dtype=torch.float32)  # batch_size=32, embedding_dim=64

# Prepare output tensor
output = torch.zeros_like(x)

# Compute self attention
gpu_kernels.self_attention(x, output, d=64)
```

## Implementation Details

### Self-Attention Process
1. **Score Computation (Q×K^T)**:
   - Tiled matrix multiplication
   - Scale by 1/√d
   - Transposed access for K

2. **Softmax**:
   - Row-wise maximum finding
   - Numerically stable exp(x - max)
   - Parallel sum reduction
   - Final normalization

3. **Value Aggregation (attention×V)**:
   - Tiled matrix multiplication
   - Regular (non-transposed) access for V

### Memory Management
- Efficient use of shared memory for tiles
- Proper CUDA memory allocation/deallocation
- Minimal memory transfers between CPU and GPU

## Testing
Run the test suite to compare with PyTorch's native implementation:
```bash
python test.py
```

The test:
- Verifies numerical accuracy
- Checks output shapes
- Validates attention patterns
- Compares with PyTorch reference

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)