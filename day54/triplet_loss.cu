#include <cuda_runtime.h>
#include <iostream>

__global__ void triplet_loss_kernel(const float* anchor, const float* positive, const float* negative, float* loss, float alpha, int n) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= n) return;
    
    float dist_positive = (anchor[idx] - positive[idx]) * (anchor[idx] - positive[idx]);
    float dist_negative = (anchor[idx] - negative[idx]) * (anchor[idx] - negative[idx]);
    
    float triplet_loss = fmaxf(0.0f, dist_positive - dist_negative + alpha);
    atomicAdd(loss, triplet_loss);
}

void triplet_loss(const float* anchor_h, const float* positive_h, const float* negative_h, float* loss_h, float alpha, int n) {
    float *anchor_d, *positive_d, *negative_d, *loss_d;
    cudaMalloc((void**)&anchor_d, n * sizeof(float));
    cudaMalloc((void**)&positive_d, n * sizeof(float));
    cudaMalloc((void**)&negative_d, n * sizeof(float));
    cudaMalloc((void**)&loss_d, sizeof(float));
    
    cudaMemcpy(anchor_d, anchor_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(positive_d, positive_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(negative_d, negative_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(loss_d, 0, sizeof(float));
    
    int blockSize = 256;
    int gridSize = ceil(n/float(blockSize));
    triplet_loss_kernel<<<gridSize, blockSize>>>(anchor_d, positive_d, negative_d, loss_d, alpha, n);
    
    cudaMemcpy(loss_h, loss_d, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(anchor_d);
    cudaFree(positive_d);
    cudaFree(negative_d);
    cudaFree(loss_d);
}
