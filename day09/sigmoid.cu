#include <cuda_runtime.h>
#include <math.h>
__global__ void sigmoid_kernel(float* input, float* output, int n){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        output[idx] = 1.0f / (1.0f+expf(-input[idx]));
    }
}

extern "C" void sigmoid(float* input_h, float* output_h, int n){
    float* input_d, *output_d;
    int size = n*sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    int blockSize = 16*16;
    int gridSize = ceil(n/float(blockSize));
    sigmoid_kernel<<<gridSize, blockSize>>>(input_d, output_d, n);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}