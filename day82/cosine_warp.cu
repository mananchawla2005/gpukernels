#include <cuda_runtime.h>
#define e 1e-8

__global__ void cosine_kernel(const float* __restrict__ P,
                                   const float* __restrict__ T,
                                   float* __restrict__ O,
                                   size_t n, size_t d)
{
    const int warpSize = 32;
    int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane         = threadIdx.x & (warpSize - 1);
    if (globalWarpId >= n) return;

    float sumP = 0, sumT = 0, dot = 0;
    for (int c = lane; c < d; c += warpSize) {
        float p = P[globalWarpId * d + c];
        float t = T[globalWarpId * d + c];
        sumP = fmaf(p, p, sumP);
        sumT = fmaf(t, t, sumT);
        dot  = fmaf(p, t, dot);
    }

    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sumP += __shfl_down_sync(0xffffffff, sumP, offset);
        sumT += __shfl_down_sync(0xffffffff, sumT, offset);
        dot  += __shfl_down_sync(0xffffffff, dot,  offset);
    }

    if (lane == 0) {
        float denom = max(1e-8f, sqrtf(sumP))*max(1e-8f, sqrtf(sumT));
        float cos  = dot/denom;
        O[globalWarpId] = 1.0f - cos;
    }
}

extern "C" void solution(const float* predictions,
                         const float* targets,
                         float*       output,
                         size_t       n,
                         size_t       d)
{
    const int warpSize       = 32;
    const int threadsPerBlock= 256;
    const int warpsPerBlock  = threadsPerBlock / warpSize;  // = 8
    int       gridSize       = (int)((n + warpsPerBlock - 1) / warpsPerBlock);

    // now each warp computes one row
    cosine_kernel<<<gridSize, threadsPerBlock>>>(
        predictions, targets, output, n, d);
}