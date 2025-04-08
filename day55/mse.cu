#include <cuda_runtime.h>
#include <iostream>

__global__ void mseKernel(const float* predictions_d, const float* targets_d, float* result_d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float diff = predictions_d[idx] - targets_d[idx];
        atomicAdd(result_d, diff * diff);
    }
}

extern "C" void solve(const float* predictions_h, const float* targets_h, int N, float &mse) {
    float *predictions_d = nullptr;
    float *targets_d = nullptr;
    float *result_d = nullptr;
    float result_h = 0.0f;
    
    cudaMalloc((void**)&predictions_d, N * sizeof(float));
    cudaMalloc((void**)&targets_d, N * sizeof(float));
    cudaMalloc((void**)&result_d, sizeof(float));
    
    cudaMemcpy(predictions_d, predictions_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(targets_d, targets_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_d, &result_h, sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    mseKernel<<<numBlocks, blockSize>>>(predictions_d, targets_d, result_d, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost);
    mse = result_h / N;
    
    cudaFree(predictions_d);
    cudaFree(targets_d);
    cudaFree(result_d);
}

int main() {
    const int N = 4;
    float predictions_h[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float targets_h[N] = {1.5f, 2.5f, 3.5f, 4.5f};
    float mse = 0.0f;
    
    solve(predictions_h, targets_h, N, mse);
    std::cout << "MSE: " << mse << std::endl;
    return 0;
}