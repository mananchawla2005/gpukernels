#include <cuda_runtime.h>
#include <math.h>
#include <cub/cub.cuh>


template <int BLOCK_SIZE>
__global__ void layer_norm_4d_kernel_vectorized(
    const float* __restrict__ X, const float* __restrict__ gamma, const float* __restrict__ beta,
    float* Y, size_t B, size_t F, size_t D1, size_t D2, float epsilon
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;

    const int tid = threadIdx.x;
    constexpr size_t vec_factor = 4;
    const size_t norm_size = F*D1*D2;
    const size_t batch_offset = batch_idx*norm_size;
    const float4* X4 = reinterpret_cast<const float4*>(X + batch_offset);
    const float4* g4 = reinterpret_cast<const float4*> (gamma);
    const float4* b4 = reinterpret_cast<const float4*> (beta);
    float4* Y4 = reinterpret_cast<float4*>(Y + batch_offset);
    size_t vec_sz = norm_size/vec_factor;

    float local_sum = 0.f;
    float local_sum_squares = 0.f;
    #pragma unroll 4
    for (size_t i = tid; i < vec_sz; i += BLOCK_SIZE) {
        float4 v = __ldg(&X4[i]);
        local_sum += v.x + v.y + v.z + v.w;
        local_sum_squares += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    // block‐reduce a pair <sum, sum_sq> note to self
    struct Pair { float s, ss; };
    struct SumOp { __device__ Pair operator()(Pair a, Pair b) const { return {a.s + b.s, a.ss + b.ss}; } };
    __shared__ typename cub::BlockReduce<Pair, BLOCK_SIZE>::TempStorage tmp;
    Pair agg = cub::BlockReduce<Pair, BLOCK_SIZE>(tmp).Reduce({local_sum,local_sum_squares}, SumOp());

    __shared__ float mean, inv_std;
    if (tid==0) {
        mean    =  agg.s / norm_size;
        float var = (agg.ss / norm_size) - mean*mean;
        inv_std = rsqrtf(var + epsilon);
    }
    __syncthreads();

    #pragma unroll 4
    for (size_t i = tid; i < vec_sz; i += BLOCK_SIZE) {
        float4 v = __ldg(&X4[i]);
        float4 G = __ldg(&g4[i]), Bv = __ldg(&b4[i]), N;
        N.x = __fmaf_rn((v.x-mean)*inv_std, G.x, Bv.x);
        N.y = __fmaf_rn((v.y-mean)*inv_std, G.y, Bv.y);
        N.z = __fmaf_rn((v.z-mean)*inv_std, G.z, Bv.z);
        N.w = __fmaf_rn((v.w-mean)*inv_std, G.w, Bv.w);
        Y4[i] = N;
    }
}

extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    constexpr int BLOCK = 1024;
    const float epsilon = 1e-5f;
    size_t shared_memory    = 0; // auto allocated note to self
    layer_norm_4d_kernel_vectorized<BLOCK><<<B, BLOCK, shared_memory>>>(X, gamma, beta, Y, B, F, D1, D2, epsilon);
}