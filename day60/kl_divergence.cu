#include <cuda_runtime.h>

__global__ void kl_divergence_kernel_optimized(
    const float* __restrict__ predictions, 
    const float* __restrict__ targets, 
    float* __restrict__ output, 
    size_t n) 
{
    const int elementsPerThread = 4; 
    const size_t baseIdx = elementsPerThread * (blockDim.x * blockIdx.x + threadIdx.x);
    const float epsilon = 1e-10f;

    #pragma unroll
    for (int i = 0; i < elementsPerThread; ++i) {
        size_t idx = baseIdx + i;
        if (idx < n) {
            float pred = predictions[idx] + epsilon;
            float target = targets[idx] + epsilon;

            float mask = (targets[idx] >= epsilon);
            float log_div = __logf(target) - __logf(pred);
            output[idx] = mask * (target * log_div); 
        }
    }
}

extern "C" void solution(
    const float* predictions, 
    const float* targets, 
    float* output, 
    size_t n) 
{
    const int threadsPerBlock = 256;
    const int elementsPerThread = 4;
    const int totalThreads = (n + elementsPerThread - 1) / elementsPerThread;
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    kl_divergence_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        predictions, targets, output, n
    );
    
    cudaDeviceSynchronize();
}
