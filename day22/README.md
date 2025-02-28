# Day 22: Implementing Batch Normalization with Triton

This project demonstrates a custom batch normalization operation using Triton for GPU acceleration. Batch normalization helps stabilize and speed up neural network training by normalizing layer inputs. In this implementation, we compute the mean and variance for every spatial location across the batch, offering a unique twist compared to PyTorch's typical per-channel normalization.

## Features

- **Custom Triton Kernel**: Implements batch normalization using Triton's grid-based parallelism.
- **PyTorch Integration**: Works seamlessly with PyTorch tensors on CUDA devices.
- **Flexible Design**: Compare results with PyTorch’s batch normalization by reshaping inputs appropriately.
- **Performance Focused**: Leverages Triton’s efficient memory management and parallel computations.

## Key Components

1. **Batch Normalization Function**:
   - `batch_norm`: Prepares inputs, launches the Triton kernel, and returns the normalized output.
2. **Triton Kernel**:
   - `batch_norm_kernel`: Computes the mean and variance for each spatial location across the batch, normalizes the data, and applies affine transformation.
3. **Testing Framework**:
   - **`test_batch_norm`**: Compares the Triton implementation with a reshaped version of PyTorch's batch norm.
   - **`test_different_sizes`**: Ensures the kernel works correctly on various input sizes.

## Prerequisites

- CUDA-capable GPU
- Python 3.8+
- PyTorch 2.0+
- Triton library

## Installation

```bash
# Create and activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# For Linux/Mac:
# python3 -m venv venv
# source venv/bin/activate

# Install dependencies
pip install torch
pip install triton
```

## Running Tests

Execute the following command in the day22 directory:

```bash
python batch_norm.py
```

If the tests pass, you should see the message "All tests passed!" printed in the console.

## References

- [Triton Documentation](https://triton-lang.org)
- [PyTorch Batch Normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [Batch Normalization Research Paper](https://arxiv.org/abs/1502.03167)
