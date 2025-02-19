#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

__global__ void gelu_kernel(float* input_d, float* output_d, int n){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<n){
        float x = input_d[idx];
        output_d[idx] = 0.5f * x * (1.0f + erff(x/sqrtf(2.0f)));
    }
}

extern "C" void gelu(float* input_h, float* output_h, int n){
    float *input_d, *output_d;
    int size = n* sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(256);
    dim3 gridSize(ceil(n/256.0));
    gelu_kernel<<<gridSize, blockSize>>>(input_d, output_d, n);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}