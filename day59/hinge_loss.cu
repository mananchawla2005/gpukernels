#include <cuda_runtime.h>

__global__ void hingeKernel(const float* predictions, const float* targets, float* output, size_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float pred = predictions[tid];
        float target = targets[tid];
        float loss = 1.0f - pred * target;
        
        output[tid] = fmaxf(0.0f, loss);
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    hingeKernel<<<gridSize, blockSize>>>(predictions, targets, output, n);
    
    cudaDeviceSynchronize();
}