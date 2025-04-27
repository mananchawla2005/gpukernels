#include <cuda_runtime.h>

__global__ void computeSumOfSquares(const float* X, float* partialSums, size_t size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (i < size) {
        sum += X[i] * X[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

__global__ void normalizeElements(const float* X, float* Y, size_t size, float frobeniusNorm) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float normFactor = (frobeniusNorm > 1e-10f) ? 1.0f / frobeniusNorm : 0.0f;
    
    while (i < size) {
        Y[i] = X[i] * normFactor;
        i += blockDim.x * gridDim.x;
    }
}

extern "C" void solution(const float* X, float* Y, size_t size) {
    const int threadsPerBlock = 256;
    const int maxBlocks = 1024;
    const int numBlocks = ((size + threadsPerBlock - 1) / threadsPerBlock);
    
    float* d_partialSums;
    cudaMalloc(&d_partialSums, numBlocks * sizeof(float));
    
    computeSumOfSquares<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(X, d_partialSums, size);
    
    float* h_partialSums = new float[numBlocks];
    cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sumOfSquares = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sumOfSquares += h_partialSums[i];
    }
    float frobeniusNorm = sqrtf(sumOfSquares);
    
    normalizeElements<<<numBlocks, threadsPerBlock>>>(X, Y, size, frobeniusNorm);
}