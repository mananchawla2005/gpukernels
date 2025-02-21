# Day 17: Custom NF4 Quantization Implementation

This project implements a custom Normal Float 4 (NF4) quantization using CUDA, similar to the one used in bitsandbytes. NF4 is a crucial component in large language model optimization, enabling significant memory reduction while maintaining model performance.

## What is NF4?
NF4 is a 4-bit quantization format that:
- Uses normalized floating-point representation
- Compresses 32-bit floats into 4 bits
- Maintains per-block scaling factors (absmax)
- Packs two 4-bit values into one byte
- Achieves 8x compression ratio

## Implementation Features
- Block-wise absmax computation using parallel reduction
- Custom decision tree for optimal quantization levels
- Efficient bit packing (2 values per byte)
- Numerically stable normalization
- Memory-efficient storage format
- PyTorch integration via pybind11

## Key Components
1. **CUDA Kernel**:
   - Parallel block-wise maximum finding
   - Decision tree for 4-bit value assignment
   - Efficient packing of two 4-bit values
   - Shared memory usage for performance

2. **PyTorch Binding**:
   - Seamless integration with PyTorch tensors
   - Proper memory management
   - Type checking and validation

3. **Python Interface**:
   - Easy-to-use quantization functions
   - Comparison with bitsandbytes implementation
   - Comprehensive testing framework

## Performance Comparison
- Original memory: 64KB (for 256x64 float32 matrix)
- Quantized memory: 8KB
- Compression ratio: 8x
- MSE loss compared to bitsandbytes: ~10^-6

## Prerequisites
- CUDA-capable GPU
- CUDA Toolkit 11.0+
- Python 3.8+
- PyTorch 2.0+
- bitsandbytes (for comparison)

## Installation
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch bitsandbytes

# Build and install the package
pip install .
```

## Usage Example
```python
import torch
import gpu_kernels

# Prepare input tensor
weight = torch.randn(256, 64, dtype=torch.float32)

# Quantize
weight_quant, absmax = gpu_kernels.quantize_nf4(
    weight.flatten(),
    block_size_outer=64,
    block_size_inner=1
)

# The weight_quant tensor contains packed 4-bit values
# absmax contains the scaling factors per block
```

## Implementation Details

### Quantization Process
1. **Block-wise Processing**:
   - Input is divided into blocks of size 64
   - Each block computes its absolute maximum
   - Values are normalized by block maximum

2. **4-bit Quantization**:
   - Uses 16 carefully chosen quantization levels
   - Decision tree for optimal level selection
   - Handles both positive and negative values

3. **Bit Packing**:
   - Two 4-bit values packed into one byte
   - MSB contains first value, LSB contains second
   - Efficient memory utilization

### Memory Layout
- Original: `float32[N]`
- Quantized: `uint8[N/2]` + `float32[N/64]`
- Additional storage for quantization codes

## Testing
Run the test suite to compare with bitsandbytes:
```bash
python test.py
```

The test:
- Compares quantization accuracy
- Verifies memory reduction
- Validates numerical stability
- Checks shape preservation

## References
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
