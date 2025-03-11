# Day 18: Double Quantization with NF4 and 8-bit

This project implements a custom double quantization scheme using NF4 (Normal Float 4) followed by 8-bit quantization for the scaling factors, inspired by QLoRA but without mean subtraction. This approach provides additional compression while maintaining reasonable accuracy.

## Double Quantization Overview

The implementation uses a two-step quantization process:
1. **Primary NF4 Quantization**: Values are first quantized to 4-bit using block-wise scaling
2. **Secondary 8-bit Quantization**: The scaling factors themselves are quantized to 8-bit

### Key Features
- Hierarchical block structure with outer and inner blocks
- Two-level scaling factors for better precision
- Efficient CUDA implementation using shared memory
- PyTorch integration for easy use with LLMs

## Implementation Details

### Quantization Process
1. **First Level (NF4)**:
   - Input tensor is divided into outer blocks (default size 128)
   - Each block computes its absolute maximum
   - Values are quantized to 4-bit using NF4 encoding
   - Two 4-bit values are packed into one byte

2. **Second Level (8-bit)**:
   - Scaling factors from first level are grouped into inner blocks (default size 32)
   - Each group gets its own scaling factor
   - These scaling factors are quantized to 8-bit

### Memory Layout
- Original: `float32[N]`
- Quantized weights: `uint8[N/2]` (packed NF4)
- Outer absmax: `uint8[N/block_size_outer]`
- Inner absmax: `float32[N/(block_size_outer*block_size_inner)]`

### Compression Ratio
For a typical layer with N elements:
- Original size: N * 4 bytes
- Quantized size: (N/2) + (N/128) + (N/4096) bytes
- Approximate compression ratio: ~7.8x

## Usage Example
```python
from gpu_kernels import quantize_nf4_double

# Prepare input tensor
weight = torch.randn(256, 64, dtype=torch.float32)

# Quantize with double quantization
weight_quant, absmax_outer, absmax_inner = quantize_nf4_double(
    weight.flatten(),
    block_size_outer=128,
    block_size_inner=32
)

# weight_quant: packed 4-bit values
# absmax_outer: 8-bit quantized scaling factors
# absmax_inner: fp32 scaling factors for absmax_outer
```

## Performance Analysis

From test results on a Llama-2 layer:
- MSE loss: ~10^-4 (slightly higher than single quantization)
- Quantization time: comparable to single quantization
- Memory savings: ~20% more than single NF4 quantization
- Minimal impact on model quality for most tasks

## Implementation Notes

1. **Block Sizes**:
   - Outer block size (128) balances precision and compression
   - Inner block size (32) optimizes scaling factor storage

2. **Precision Considerations**:
   - No mean subtraction (unlike QLoRA)
   - Uses absolute maximum for scaling
   - Maintains separate inner scaling factors in fp32

3. **CUDA Optimization**:
   - Shared memory for parallel reduction
   - Coalesced memory access patterns
   - Efficient bit packing operations

## Prerequisites
- CUDA-capable GPU
- CUDA Toolkit 11.0+
- Python 3.8+
- PyTorch 2.0+

## Testing
```bash
python test.py
```

The test script:
- Loads a Llama model layer
- Applies double quantization
- Measures compression ratios
- Calculates MSE loss
- Reports timing information

## References
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)