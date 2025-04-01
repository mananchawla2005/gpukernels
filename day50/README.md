# Day 50: Self-Attention Backward Pass in CUDA

This project implements an efficient CUDA-based backward pass for self-attention, a critical component in transformer architectures. This implementation marks the halfway point of a 100-day GPU kernels journey.

## What is Self-Attention Backward Pass?

The backward pass for self-attention computes the gradients needed for training transformer models:
- Propagates gradients through the attention mechanism
- Calculates gradients for query (Q), key (K), and value (V) matrices
- Handles the chain rule through softmax and matrix multiplications
- Ensures numerical stability throughout the process

## Implementation Features

- Complete backward pass algorithm for self-attention
- Tiled matrix multiplication for efficient gradient computation
- Numerically stable softmax backward pass
- Proper scaling factor handling (1/√d)
- PyTorch integration via pybind11

## Key Components

1. **CUDA Kernels**:
   - `softmax_backward_kernel`: Computes gradients through softmax operation
   - `tiled_matmul_kernel`: Efficient matrix multiplication with transpose support
   - Support for various matrix operations required in backpropagation
   - Memory-efficient implementation

2. **PyTorch Binding**:
   - Seamless integration with PyTorch's autograd
   - Proper CUDA memory management
   - Type checking and validation
   - Device transfer handling

3. **Python Interface**:
   - Simple API for backward pass computation
   - Compatible with PyTorch tensors
   - Easy comparison with PyTorch's native implementation

## Prerequisites

- CUDA-capable GPU
- CUDA Toolkit 11.0+
- Python 3.8+
- PyTorch 2.0+

## Setup

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


## Implementation Details

### Backward Pass Algorithm

1. **Value Gradient (dV)**:
   - Compute dV = P^T × dO
   - Transpose attention weights and multiply with output gradients

2. **Attention Weights Gradient (dP)**:
   - Compute dP = dO × V^T
   - Multiply output gradients with transposed value matrix

3. **Softmax Backward**:
   - Compute dS = P ⊙ (dP - sum(dP ⊙ P))
   - Numerically stable implementation using shared memory

4. **Query and Key Gradients**:
   - Compute dQ = (dS × K)/√d
   - Compute dK = (dS^T × Q)/√d
   - Apply proper scaling factor

### Memory Management

- Efficient use of shared memory for tiles
- Proper CUDA memory allocation/deallocation
- Minimal memory transfers between CPU and GPU

## Testing

Run the test script to verify the implementation:

```bash
python test.py
```

The test:
- Compares gradients with PyTorch's autograd
- Verifies numerical accuracy within tolerance
- Validates gradient shapes and patterns
- Confirms the scaling factor is applied correctly

## Special Note

This implementation marks the halfway point (Day 50) of a 100-day journey exploring GPU kernel development. It builds upon the forward pass implementation from Day 19 to complete a fully functional self-attention mechanism with both forward and backward passes.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Backpropagation Through Softmax](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)