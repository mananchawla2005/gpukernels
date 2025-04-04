#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void l1NormAtomicKernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(output, fabsf(input[i]));
    }
}

float l1NormAtomic(const float* input_h, int n) {
    float *input_d, *output_d;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    cudaMalloc(&input_d, n * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));
    cudaMemset(output_d, 0, sizeof(float));
    cudaMemcpy(input_d, input_h, n * sizeof(float), cudaMemcpyHostToDevice);
    l1NormAtomicKernel<<<numBlocks, blockSize>>>(input_d, output_d, n);
    cudaDeviceSynchronize();
    float norm;
    cudaMemcpy(&norm, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(input_d);
    cudaFree(output_d);
    
    return norm;
}

int main() {
    const int N = 1000000;
    float* input_h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        input_h[i] = 1.0f;
    }
    float norm = l1NormAtomic(input_h, N);
    printf("L1 norm of vector using atomic: %f\n", norm);
    printf("Expected value: %f\n", (float)N);
    free(input_h);
    return 0;
}