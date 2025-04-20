#include <cuda_runtime.h>

__global__ void solution_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; // column index
    const int ty = blockIdx.y * blockDim.y + threadIdx.y; // row index
    
    if (tx < m/4 && ty < n) {
        const int idx = ty * (m/4) + tx;
        
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + idx);
        
        v.x = 1 / (1+__expf(-v.x));
        v.y = 1 / (1+__expf(-v.y));
        v.z = 1 / (1+__expf(-v.z));
        v.w = 1 / (1+__expf(-v.w));
        
        reinterpret_cast<float4*>(output)[idx] = v;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    
    dim3 blockDim(128, 8); 
    dim3 gridDim((m/4 + blockDim.x - 1) / blockDim.x, 
                 (n + blockDim.y - 1) / blockDim.y);
    
    solution_kernel<<<gridDim, blockDim>>>(input, output, n, m);
}

