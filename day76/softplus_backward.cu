#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
static __device__ __forceinline__ T softplus_backward(T x) {
    // Derivative of softplus is sigmoid: e^x / (1 + e^x)
    return __expf(x) / (1 + __expf(x));
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

        v.x = softplus_backward(v.x);
        v.y = softplus_backward(v.y);
        v.z = softplus_backward(v.z);
        v.w = softplus_backward(v.w);

        output[idx] = v;
    }
}
