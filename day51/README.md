# Day 51: Lightning Attention Implementation

This project implements Lightning Attention, an efficient attention mechanism that improves upon standard attention implementations by decomposing the attention computation into intra-block and inter-block components, reducing computational complexity and enabling processing of longer sequences.

## What is Lightning Attention?

Lightning Attention is an optimization technique that:

1. Decomposes the attention calculation into two components: intra-block and inter-block attention
2. Reduces computational complexity from O(N²) to approximately O(Nd^2)
3. Improves memory efficiency by avoiding storage of the full attention matrix
4. Maintains comparable accuracy to standard attention while being more efficient
5. Supports causal masking for autoregressive models

## How It Works

### Key Decomposition Strategy

Standard attention requires computing the full N×N attention matrix. Lightning Attention instead:

- Divides input matrices (Q, K, V) into blocks of size B×D
- Computes two separate components:
  - Intra-block: detailed attention within each block
  - Inter-block: summarized attention between blocks
- Combines both components for the final attention output
- Uses a causal mask to ensure tokens only attend to previous tokens

### Lightning Attention Algorithm

The implementation consists of two main components:

1. **Intra-Block Attention (O_Intra)**:
   - Computes standard attention within blocks
   - Applies causal masking to prevent attending to future tokens
   - Formula: O_Intra = [(Q·K^T)·M]·V where M is the causal mask

2. **Inter-Block Attention (O_Inter)**:
   - Pre-computes the KV product to summarize information across blocks
   - Uses this summary to compute attention between blocks
   - Formula: O_Inter = Q·(K·V)

3. **Final Output**:
   - Combines both components: O = (O_Intra + O_Inter)/√d
   - Applies normalization to maintain stable output magnitude

### Key Optimizations

1. **Computational Efficiency**:
   - Reduces operations from O(N²) to approximately O(N√N)
   - Avoids redundant computations through matrix factorization
   - Uses shared memory for faster data access

2. **Memory Efficiency**:
   - Avoids storing the full N×N attention matrix
   - Reuses shared memory for different computation stages
   - Uses atomic operations to safely accumulate results in parallel

3. **Performance**:
   - Uses shared memory to reduce global memory access
   - Optimized thread block organization
   - Efficient causal masking implementation

## Implementation Details

The implementation uses:
- Fixed block size B = 16 for optimal performance
- Embedding dimension D = 64 (can be adjusted)
- Shared memory for efficient data access and computation
- Automatic causal masking for autoregressive models
- CUDA kernel optimized for parallel computation
- PyBind11 for seamless Python integration

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
pip install -e .
```

## Testing and Usage Example
Run the test script to verify the implementation:

```bash
python test.py
```

Example usage in your own code:

```python
import torch
import gpu_kernels

# Create input tensors
n = 64  # sequence length
d = 64  # embedding dimension
q = torch.rand((n, d), dtype=torch.float32)  # query matrix
k = torch.rand((n, d), dtype=torch.float32)  # key matrix
v = torch.rand((n, d), dtype=torch.float32)  # value matrix

# Prepare output tensor
output = torch.zeros_like(q)

# Run Lightning Attention
gpu_kernels.lightning_attention(q, k, v, output)
```

## Performance Analysis

Lightning Attention provides significant benefits over standard attention:

1. **Computational Efficiency**: Reduces computation from O(N²) to approximately O(N√N)
2. **Memory Usage**: Requires less memory than standard attention implementations
3. **Scaling**: Handles longer sequences more efficiently
4. **Causal Support**: Built-in support for causal masking in autoregressive models

## External References

[Lightning Attention Paper](https://arxiv.org/pdf/2405.17381.pdf)  
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
[PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
