#include <cuda_runtime.h>


__global__ void solution_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m, const float alpha, size_t m4) {
    
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ty < n && tx < m4) {
        const int idx = ty * m4 + tx;        
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + idx);;
        const float factor = alpha - 1.0f;
        
        v.x = (__expf(2*v.x)-1)/(__expf(2*v.x)+1);
        v.y = (__expf(2*v.y)-1)/(__expf(2*v.y)+1);
        v.z = (__expf(2*v.z)-1)/(__expf(2*v.z)+1);
        v.w = (__expf(2*v.w)-1)/(__expf(2*v.w)+1);
        
        reinterpret_cast<float4*>(output)[idx] = v;
    }
}


// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {    
    
    dim3 blockDim(32, 16); 
    dim3 gridDim((m/4 + blockDim.x - 1) / blockDim.x, 
                 (n + blockDim.y - 1) / blockDim.y);
    
    solution_kernel<<<gridDim, blockDim>>>(input, output, n, m, alpha, m/4);
}