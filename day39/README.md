# Day 39:
# TASK: Coarsened Tiled Matrix Multiplication

This implementation enhances the standard tiled matrix multiplication approach with thread coarsening to improve computational efficiency and memory access patterns.

## What is Coarsened Matrix Multiplication?

Coarsened matrix multiplication is an optimization technique that:

1. Assigns multiple output elements to each thread (coarsening factor of 8 in this implementation)
2. Reduces thread scheduling overhead by increasing work per thread
3. Improves instruction-level parallelism within each thread
4. Achieves approximately 5% performance improvement over standard tiled matmul

## How It Works

### Thread Coarsening

Standard tiled matrix multiplication assigns each thread to compute a single output element. With thread coarsening:

- Each thread computes multiple (8) output elements along a row
- The thread reuses loaded M matrix values for multiple computations
- Grid dimensions are adjusted to account for the coarsening factor
- Each thread maintains multiple partial products in registers

### Key Optimizations

1. **Efficient Register Usage**:
   - Multiple partial products are stored in registers (`Pval` array)
   - Values from M matrix are loaded once and reused
   - Reduces shared memory bank conflicts

2. **Improved Memory Access**:
   - One thread computes multiple consecutive elements along a row
   - Better memory coalescing for output writes
   - More compute-intense kernels that better hide memory latency

3. **Grid Configuration**:
   - Adjusted grid size to account for coarsening factor
   - Each thread block processes `TILE_WIDTH × (TILE_WIDTH * COARSE_FACTOR)` elements

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
pip install .
```

## Testing and Usage Example
Run the test script to verify the installation:

```bash
python test.py
```

- The script compares the GPU implementation against PyTorch's CPU implementation
- For small matrices, it prints the results for visual verification

**Note**: Ensure you're in the activated virtual environment when running the code.

## Performance Analysis

Coarsened matmul typically provides around 5% performance improvement over standard tiled matmul due to:

1. **Higher Arithmetic Intensity**: More computation per memory access
2. **Better Instruction-Level Parallelism**: Multiple independent operations per thread
3. **Reduced Thread Management Overhead**: Fewer total threads to schedule
4. **More Efficient Register Usage**: Better utilization of register file

## Implementation Details

The implementation uses:
- `TILE_WIDTH` of 32 for optimal shared memory usage
- `COARSE_FACTOR` of 8 for balance between parallelism and resource usage
- Shared memory for tile storage
- CUDA events for performance timing
- PyBind11 for Python bindings

## External References

[Matrix Multiplication Performance Optimizations](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
[PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
