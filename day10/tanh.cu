#include <cuda_runtime.h>

__global__ void tanh_kernel(float* input, float* output, int n){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        float ex = expf(input[idx]);
        float emx = expf(-input[idx]);
        output[idx] = (ex-emx)/(ex+emx);
    }
    
}

extern "C" void naive_tanh(float* input_h, float* output_h, int n){
    float* input_d, *output_d;
    int size = n*sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    int blockSize = 16*16;
    int gridSize = ceil(n/float(blockSize));
    tanh_kernel<<<gridSize, blockSize>>>(input_d, output_d, n);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}