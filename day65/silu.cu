#include<cuda_runtime.h>

__constant__ float lambda = 1.0507f;
__constant__ float alpha = 1.67326;


__global__ void solution_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; // column index
    const int ty = blockIdx.y * blockDim.y + threadIdx.y; // row index
    
    if (tx < m/4 && ty < n) {
        const int idx = ty * (m/4) + tx;
        
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + idx);
        
        v.x = lambda*(fmaxf(v.x, 0.0f)+fminf(0, __fmaf_rn(alpha, __expf(v.x), -alpha)));
        v.y = lambda*(fmaxf(v.y, 0.0f)+fminf(0, __fmaf_rn(alpha, __expf(v.y), -alpha)));
        v.z = lambda*(fmaxf(v.z, 0.0f)+fminf(0, __fmaf_rn(alpha, __expf(v.z), -alpha)));
        v.w = lambda*(fmaxf(v.w, 0.0f)+fminf(0, __fmaf_rn(alpha, __expf(v.w), -alpha)));
        
        reinterpret_cast<float4*>(output)[idx] = v;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    
    dim3 blockDim(32, 16); 
    dim3 gridDim((m/4 + blockDim.x - 1) / blockDim.x, 
                 (n + blockDim.y - 1) / blockDim.y);
    
    solution_kernel<<<gridDim, blockDim>>>(input, output, n, m);
}