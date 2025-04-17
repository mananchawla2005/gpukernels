#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_4d_kernel_vectorized(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* Y,
    size_t B, size_t F, size_t D1, size_t D2,
    float epsilon
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const size_t norm_size = F * D1 * D2;
    const size_t batch_offset = batch_idx * norm_size;

    extern __shared__ float shared_mem[];
    float* sum_shared = shared_mem;
    float* sum_squares_shared = shared_mem + block_size;

    // Vectorized pointer
    const float4* X4 = reinterpret_cast<const float4*>(X + batch_offset);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    float4* Y4 = reinterpret_cast<float4*>(Y + batch_offset);

    size_t vec_size = norm_size / 4;

    float4 tmp;
    float local_sum = 0.0f;
    float local_sum_squares = 0.0f;

    #pragma unroll 4
    for (size_t i = tid; i < vec_size; i += block_size) {
        tmp = __ldg(&X4[i]);
        
        local_sum += tmp.x + tmp.y + tmp.z + tmp.w;
        local_sum_squares += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }

    sum_shared[tid] = local_sum;
    sum_squares_shared[tid] = local_sum_squares;
    __syncthreads();

    #pragma unroll 4
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
            sum_squares_shared[tid] += sum_squares_shared[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float mean, variance;
    if (tid == 0) {
        mean = sum_shared[0] / norm_size;
        variance = (sum_squares_shared[0] / norm_size) - (mean * mean);
    }
    __syncthreads();

    float stddev = sqrtf(variance + epsilon);

    #pragma unroll 4
    for (size_t i = tid; i < vec_size; i += block_size) {
        tmp = __ldg(&X4[i]);
        float4 g = __ldg(&gamma4[i]);
        float4 b = __ldg(&beta4[i]);

        float4 norm;
        norm.x = (tmp.x - mean) / stddev * g.x + b.x;
        norm.y = (tmp.y - mean) / stddev * g.y + b.y;
        norm.z = (tmp.z - mean) / stddev * g.z + b.z;
        norm.w = (tmp.w - mean) / stddev * g.w + b.w;

        Y4[i] = norm;
    }
}

extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    const float epsilon = 1e-5f;
    int block_size = 256;
    int grid_size = B;
    size_t shared_mem_size = block_size * sizeof(float) * 2;
    layer_norm_4d_kernel_vectorized<<<grid_size, block_size, shared_mem_size>>>(
        X, gamma, beta, Y, B, F, D1, D2, epsilon
    );
}