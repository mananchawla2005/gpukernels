#include <cuda_runtime.h>
__global__ void fused_add_relu_kernel(float* __restrict__ output, const float* __restrict__ input1, const float* __restrict__ input2, const int n){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<n){
        float sum = input1[idx]+input2[idx];
        output[idx] = sum > 0 ? sum : 0;
    }
}

extern "C" void fused_add_relu(float* output_h, float* input1_h, float* input2_h, const int n){
    float* output_d, *input1_d, *input2_d;
    int size = n * sizeof(float);
    cudaMalloc((void**)&output_d, size);
    cudaMalloc((void**)&input1_d, size);
    cudaMalloc((void**)&input2_d, size);
    cudaMemcpy(input1_d, input1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(input2_d, input2_h, size, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    fused_add_relu_kernel<<<gridSize, blockSize>>>(output_d, input1_d, input2_d, n);
    cudaDeviceSynchronize();
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(output_d);
    cudaFree(input1_d);
    cudaFree(input2_d);
}