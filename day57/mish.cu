#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void mish_kernel(float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float softplus = logf(1 + expf(val)); 
        y[idx] = val * tanhf(softplus);       
    }
}

void mish_cuda(float* x_d, float* y_d, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    mish_kernel<<<blocks_per_grid, threads_per_block>>>(x_d, y_d, size);
    cudaDeviceSynchronize();
}