# Day 20: Introduction to Triton Programming

This project demonstrates basic Triton programming concepts for GPU acceleration, implementing tensor operations and matrix multiplication using Triton's high-level abstractions. Triton provides a Python-based approach to writing efficient GPU kernels.

## What is Triton?
Triton is a language and compiler for writing highly efficient custom Deep Learning primitives that:
- Provides Python-based GPU programming
- Enables automatic optimization of GPU kernels
- Simplifies CUDA programming concepts
- Integrates seamlessly with PyTorch

## Implementation Features
- Basic tensor operations using Triton kernels
- Efficient matrix multiplication implementation
- Block-based computation patterns
- PyTorch integration and validation

## Key Components
1. **Basic Operations**:
   - `tensor_copy_kernel`: Simple tensor copy operation
   - `check_tensors_gpu_ready`: Validation of tensor properties
   - Support for contiguous GPU tensors
   - Memory-efficient implementations

2. **Matrix Multiplication**:
   - `matmul_kernel`: Blocked matrix multiplication
   - Efficient use of Triton's dot operation
   - Stride-aware tensor access
   - Configurable block sizes

3. **Utility Functions**:
   - Ceiling division helper
   - Tensor validation
   - Grid computation helpers
   - PyTorch integration utilities

## Performance Features
- Block-based computation for better memory locality
- Efficient memory access patterns
- Automatic optimization through Triton
- Compatible with PyTorch's native operations

## Prerequisites
- CUDA-capable GPU
- Python 3.8+
- PyTorch 2.0+
- Triton library

## Installation
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch
pip install triton
```

## Usage Example
```python
import torch
import triton
import triton.language as tl

# Simple tensor copy
x = torch.tensor([1,2,3,4,5,6]).to('cuda')
z = tensor_copy(x, bs=2, fn=tensor_copy_kernel)

# Matrix multiplication
a = torch.randn(128, 256, device='cuda')
b = torch.randn(256, 64, device='cuda')
c = matmul(a, b)
```

## Implementation Details

### Tensor Copy Operation
1. **Basic Structure**:
   - Block-based processing
   - Efficient memory access
   - Mask-based boundary handling

2. **Matrix Multiplication**:
   - Block-tiled implementation
   - Efficient use of Triton's dot product
   - Stride-aware memory access
   - Boundary condition handling

### Memory Management
- Automatic memory handling through Triton
- Efficient block-based access patterns
- Integration with PyTorch's memory model

## Testing
Run the implementation with:
```bash
python triton_basics.py
```

The code:
- Implements basic tensor operations
- Demonstrates matrix multiplication
- Validates against PyTorch results
- Includes numerical accuracy checks

## References
- [Triton Documentation](https://triton-lang.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)