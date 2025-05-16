#include <cuda_runtime.h>
#include <math_functions.h> 

__global__ void adam_kernel(
    float* __restrict__ params, 
    const float* __restrict__ grads,
    float* __restrict__ m, 
    float* __restrict__ v,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int t, int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Precompute bias correction terms once per block
    __shared__ float b1_correction, b2_correction;
    if (threadIdx.x == 0) {
        b1_correction = 1.0f - __powf(beta1, t);
        b2_correction = 1.0f - __powf(beta2, t);
    }
    __syncthreads();

    float grad = grads[idx];
    
    // Fused weight decay
    grad = weight_decay != 0.0f ? 
           grad + weight_decay * params[idx] : 
           grad;

    // Update moments with fused multiply-add
    m[idx] = __fmaf_rn(beta1, m[idx], (1.0f - beta1) * grad);
    v[idx] = __fmaf_rn(beta2, v[idx], (1.0f - beta2) * grad * grad);

    // Fast approximate math functions
    float m_hat = m[idx] / b1_correction;
    float v_hat = v[idx] / b2_correction;
    float denom = __fsqrt_rn(v_hat) + eps;

    // Final parameter update
    params[idx] -= lr * m_hat / denom;
}

void launch_adam(
    float* params, float* grads,
    float* m, float* v,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int t, int n
) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    adam_kernel<<<gridSize, blockSize>>>(
        params, grads, m, v,
        lr, beta1, beta2, eps,
        weight_decay, t, n
    );
    
    cudaDeviceSynchronize();
}