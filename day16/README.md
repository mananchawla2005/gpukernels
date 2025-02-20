# Day 16:
# TASK: CUDA Device Query Implementation

This task implements a comprehensive CUDA device query tool that provides detailed information about available NVIDIA GPUs in the system. This tool is essential for CUDA development and debugging.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler compatible with your CUDA version

## WHY GPU QUERYING?

GPU querying is important for several reasons:
- Understanding hardware capabilities
- Optimizing CUDA kernel parameters
- Debugging device-specific issues
- Planning resource allocation
- Ensuring compatibility with required features

## Implementation Features
- Detailed hardware specifications
- Memory hierarchy information
- Thread and block limitations
- Hardware architecture details
- Feature support detection

## Query Information Categories
Our implementation displays:
- Basic device information
- Compute capability
- Memory specifications
- Thread/Block configurations
- Hardware architecture details
- Feature support status

## Setup
1. Compile the CUDA program:
```bash
nvcc query.cu -o query
```

2. Run the executable:
```bash
./query
```

## Implementation Details
Our query tool includes:
- Device count detection
- Error handling
- Memory specifications
  - Global memory size
  - Memory clock rates
  - Memory bus width
  - Cache sizes
- Thread/Block specifications
  - Maximum threads per block
  - Thread dimensions
  - Grid dimensions
- Hardware details
  - Number of SMs
  - Clock rates
  - Warp size
- Feature support
  - Concurrent kernel execution
  - ECC support
  - Unified addressing

## Sample Output
```
====================================
CUDA Device Query
====================================

Device 0: "NVIDIA GeForce RTX 3050 Laptop GPU"
------------------------------------
CUDA Capability Major/Minor version: 8.6
Total Global Memory: 4.00 GB
Memory Clock Rate: 5501 MHz
Memory Bus Width: 128-bit
...
```

## Usage Examples
1. Basic device information:
```bash
./query
```

2. Redirecting output to a file:
```bash
./query > gpu_info.txt
```

## External References

[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
[NVIDIA Developer Blog](https://developer.nvidia.com/blog)