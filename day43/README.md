# Day 43: Flash Attention Implementation

This project implements Flash Attention, a memory-efficient attention algorithm that reduces the memory complexity of standard attention from O(N²) to O(N), making it possible to process longer sequences on limited GPU memory.

## What is Flash Attention?

Flash Attention is an optimization technique that:

1. Processes attention in blocks to maximize data reuse from GPU on-chip SRAM
2. Avoids storing the large N×N attention matrix in global memory
3. Uses a progressive softmax algorithm to compute attention incrementally
4. Achieves significant speedup (2-4×) over standard attention implementations
5. Enables longer context lengths with the same memory budget

## How It Works

### Tiled Processing

Standard attention computes the full attention matrix at once, requiring O(N²) memory. Flash Attention instead:

- Divides the input matrices (Q, K, V) into tiles of size TILE_SIZE×D
- Processes one Q tile against all K/V tiles sequentially
- Uses shared memory to store the current tiles being processed
- Accumulates results progressively to maintain correctness

### Progressive Softmax Algorithm

The key innovation in Flash Attention is computing softmax progressively:

- For each new tile, computes local maximum and sum
- Combines with previous results using a stable algorithm
- Updates the running output based on the new normalization factors
- Maintains numerical stability with appropriate scaling

### Key Optimizations

1. **Memory Efficiency**:
   - Avoids storing the full N×N attention matrix
   - Reuses shared memory for different tiles
   - Only keeps O(N) intermediate results in memory

2. **Numerical Stability**:
   - Tracks running maximum values for stable exponentiation
   - Updates softmax normalization incrementally
   - Properly scales intermediate results

3. **Performance**:
   - Uses shared memory for faster access to current tiles
   - Reduces memory transfers between GPU and global memory
   - Enables processing of longer sequences

## Numerical Example

Consider a simplified case with:
- Sequence length N = 4
- Embedding dimension D = 2
- TILE_SIZE = 2

**Input:**
- Q = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
- K = [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]]
- V = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

**Processing:**

**Iteration 1: Q tile 1 (rows 0-1) with K/V tile 1 (rows 0-1)**
1. Load Q[0:2], K[0:2], V[0:2] to shared memory
2. Compute partial attention scores:
   - S[0,0] = (1.0×0.5 + 2.0×1.5)/√2 = 1.77
   - S[0,1] = (1.0×2.5 + 2.0×3.5)/√2 = 3.89
   - S[1,0] = (3.0×0.5 + 4.0×1.5)/√2 = 4.95
   - S[1,1] = (3.0×2.5 + 4.0×3.5)/√2 = 10.61
3. Find max values: m₁[0] = 3.89, m₁[1] = 10.61
4. Compute exp(S - m₁) and sum:
   - P[0,0] = exp(1.77-3.89) = 0.12, P[0,1] = exp(3.89-3.89) = 1.0
   - P[1,0] = exp(4.95-10.61) = 0.004, P[1,1] = exp(10.61-10.61) = 1.0
   - l₁[0] = 0.12+1.0 = 1.12, l₁[1] = 0.004+1.0 = 1.004
5. Update partial output:
   - O₁[0] = (0.12×[0.1,0.2] + 1.0×[0.3,0.4])/1.12 = [0.28,0.38]
   - O₁[1] = (0.004×[0.1,0.2] + 1.0×[0.3,0.4])/1.004 = [0.30,0.40]

**Iteration 2: Q tile 1 with K/V tile 2 (rows 2-3)**
(Similar calculation updating previous results)

**Final Output:**
After processing all tiles, we get the complete attention output O.

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

# Run Flash Attention
gpu_kernels.flash_attention(q, k, v, output)
```

## Performance Analysis

Flash Attention provides significant benefits over standard attention:

1. **Memory Efficiency**: Reduces memory usage from O(N²) to O(N), enabling longer sequences
2. **Speed**: Generally 2-4× faster than standard attention implementations
3. **Scaling**: Performance advantage increases with sequence length
4. **Numerically Equivalent**: Produces the same mathematical result as standard attention

## Implementation Details

The implementation uses:
- `TILE_SIZE` of 16 for optimal shared memory usage
- `D` fixed at 64 for demonstration purposes (embedding dimension)
- Shared memory for efficient tile storage and computation
- CUDA events for performance timing
- Thread block organization aligned with tile dimensions
- PyBind11 for seamless Python integration

## External References

[Flash Attention Paper](https://arxiv.org/abs/2205.14135)  
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
[PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)