#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
static __device__ __forceinline__ T softplus(T x) {
    return __logf(1+__expf(x));
}

__global__ void solution_kernel(const float4* __restrict__ input,
                                float4*       __restrict__ output,
                                size_t        n4,  
                                size_t        m4)  
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx = ty * m4 + tx;

    if (tx < m4 && ty < n4) {
        float4 v = __ldg(input + idx);

        v.x = softplus(v.x);
        v.y = softplus(v.y);
        v.z = softplus(v.z);
        v.w = softplus(v.w);

        output[idx] = v;
    }
}

extern "C" void solution(const float* input_f,
                         float*       output_f,
                         size_t       n,
                         size_t       m)
{
    const float4* input  = reinterpret_cast<const float4*>(input_f);
    float4*       output = reinterpret_cast<float4*>(output_f);

    size_t m4 = m / 4;
    size_t n4 = n;

    dim3 blockDim, gridDim;
    if      (n*m <= 4096ULL*4096ULL) { blockDim = {32,16,1}; }
    else if (n*m <= 8192ULL*4096ULL) { blockDim = {64, 8,1}; }
    else                              { blockDim = {128,4,1}; }

    gridDim.x = (m4 + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n4 + blockDim.y - 1) / blockDim.y;
    gridDim.z = 1;

    solution_kernel<<<gridDim, blockDim>>>(input, output, n4, m4);
}
