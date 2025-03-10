# Day 31: Corner-Turned Tiled Matrix Multiplication

This implementation demonstrates an optimized tiled matrix multiplication approach using corner turning to reduce shared memory bank conflicts and improve performance.

## What is Corner-Turned Matrix Multiplication?

Corner-turned matrix multiplication is an optimization technique that:

1. Improves shared memory access patterns during computation
2. Reduces bank conflicts by rearranging how tiles are loaded into shared memory
3. Trades slightly less efficient global memory loads for much more efficient shared memory access
4. Can provide 10-30% performance improvement over standard tiled matmul for large matrices

## How It Works

### Shared Memory Bank Conflicts

GPUs organize shared memory into multiple memory banks (typically 32 banks on modern NVIDIA GPUs). When multiple threads in a warp access different addresses in the same bank, these accesses are serialized, causing performance degradation.

- **Organization**: Shared memory is divided into equally-sized memory banks
- **Addressing**: Consecutive 32-bit words are assigned to consecutive banks
- **Parallel Access**: All banks can be accessed simultaneously when there are no conflicts

### Example of Bank Conflict

Assume shared memory has 32 banks and each bank handles 4-byte (32-bit) values:

```
Bank 0:  addresses 0, 32, 64, ...
Bank 1:  addresses 4, 36, 68, ...
Bank 2:  addresses 8, 40, 72, ...
...and so on
```

If threads in a warp access:
- Thread 0: address 0 (Bank 0)
- Thread 1: address 32 (Bank 0 again!) ← CONFLICT!
- Thread 2: address 64 (Bank 0 again!) ← CONFLICT!

These accesses will be serialized (executed one after another), significantly slowing down execution.

### The Corner Turning Solution

In the standard tiled matmul approach, threads in a warp may frequently access the same bank during the dot product calculation phase:

```cuda
// Standard approach (prone to bank conflicts)
Nds[ty][tx] = N[n_row * width + col];

// During multiplication:
Pval += Mds[ty][i] * Nds[i][tx];  // Threads with same tx access same bank
```

With corner turning, we swap indices when loading data to optimize memory access patterns:

```cuda
// Corner-turned approach (reduces bank conflicts)
Nds[tx][ty] = N[col * width + n_row];

// During multiplication:
Pval += Mds[ty][i] * Nds[i][tx];  // Threads now access different banks
```

### Visual Explanation

**Standard layout (with conflicts):**
```
Thread 0 accesses: Nds[0][0], Nds[1][0], Nds[2][0]... (same bank)
Thread 1 accesses: Nds[0][1], Nds[1][1], Nds[2][1]... (same bank)
```

**Corner-turned layout (no conflicts):**
```
Thread 0 accesses: Nds[0][0], Nds[1][0], Nds[2][0]... (different banks)
Thread 1 accesses: Nds[0][1], Nds[1][1], Nds[2][1]... (different banks)
```

Note that `Mds` naturally avoids bank conflicts because each thread accesses consecutive memory locations in a row, which maps to different banks:

```
Thread 0: Mds[0][0], Mds[0][1], Mds[0][2]... (consecutive memory)
Thread 1: Mds[1][0], Mds[1][1], Mds[1][2]... (consecutive memory)
```

## Performance Benefits

1. **Reduced Bank Conflicts**: By transforming the data layout in shared memory, threads access different banks during multiplication
2. **Improved Parallelism**: Multiple memory accesses can happen simultaneously instead of being serialized
3. **Better Memory Throughput**: Overall memory bandwidth utilization is higher
4. **Computation Efficiency**: The inner dot product loop runs with fewer stalls

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## Setup and Installation

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

## References

- [NVIDIA Shared Memory Bank Conflicts](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Matrix Multiplication Optimizations](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [Memory Coalescing in CUDA](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)