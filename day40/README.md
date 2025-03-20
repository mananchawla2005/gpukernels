# Day 40: CUDA Matrix Multiplication Profiling with Nsight Compute

This module demonstrates how to compile, execute, and profile CUDA kernels using NVIDIA's advanced profiling tools, focusing on our optimized coarsened tiled matrix multiplication implementation.

## What is Nsight Compute?

Nsight Compute is NVIDIA's comprehensive profiling tool for CUDA applications that:

1. Provides detailed kernel performance metrics and optimization suggestions
2. Visualizes memory access patterns, instruction throughput, and resource utilization
3. Helps identify performance bottlenecks in GPU workloads
4. Supports interactive and command-line profiling

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+ installed (includes nvcc compiler)
- NVIDIA Nsight Compute (included with CUDA Toolkit)
- A compatible C/C++ compiler

## Compilation Instructions

1. Create a main function to invoke your kernel:

```bash
# Create a main.cu file that calls our kernel
nvcc tiledmm_candidate_for_profiling.cu -o tiledmm_candidate_for_profiling
```

2. Compile with appropriate flags:
   - `-O3`: Enable aggressive optimization
   - `-arch=sm_75`: Set target architecture (replace with your GPU's compute capability)
   - `-lineinfo`: (Optional) Add source line information for better profiling
   - `-G`: (Optional) Include debug information (disables optimizations)

## Profiling with Nsight Compute CLI

To profile the executable using the command line:

```bash
ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_global_ld.sum,sm__sass_thread_inst_executed_op_global_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum -o profile_report ./tiledmm_candidate_for_profiling 1024
```

Key metrics collected:
- Execution time duration
- SM (Streaming Multiprocessor) utilization
- Memory throughput percentage
- Global memory load/store instructions
- L1 cache sector access counts

## Profiling with Nsight Compute GUI

For interactive profiling and visualization:

```bash
ncu --gui ./tiledmm_candidate_for_profiling 1024
```

To view a saved profiling report:

```bash
ncu-ui profile_report.ncu-rep
```

## Optimizing Based on Profile Results

Common optimization strategies based on profiling insights:

1. **Memory-Bound Kernels**:
   - Review global memory access patterns
   - Consider memory access coalescing improvements
   - Evaluate shared memory usage efficiency

2. **Compute-Bound Kernels**:
   - Look for instruction throughput bottlenecks
   - Consider loop unrolling or instruction-level optimizations
   - Check for thread divergence in your kernel

3. **Resource-Limited Kernels**:
   - Examine register usage per thread
   - Review shared memory allocation
   - Consider adjusting block dimensions or coarsening factor

## Analyzing Our Coarsened Matrix Multiplication

The matrix multiplication kernel can be analyzed for:

1. **Occupancy**: How effectively are SMs utilized?
2. **Memory Efficiency**: What percentage of theoretical bandwidth is achieved?
3. **Arithmetic Intensity**: What is the ratio of compute operations to memory operations?
4. **Register Pressure**: Is register usage limiting occupancy?


## External References

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/index.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Matrix Multiplication Optimization Techniques](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [NVIDIA Performance Analysis Tools](https://developer.nvidia.com/performance-analysis-tools)