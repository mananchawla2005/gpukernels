#include <cuda_runtime.h>

__global__ void hard_sigmoid_backward_kernel(const float* __restrict__ grad_output, 
                                            const float* __restrict__ input, 
                                            float* __restrict__ grad_input, 
                                            size_t n, size_t m)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx < m/4 && ty < n) {
        const int idx = ty * (m/4) + tx;
        
        float4 in_val = __ldg(reinterpret_cast<const float4*>(input) + idx);
        
        float4 grad_out = __ldg(reinterpret_cast<const float4*>(grad_output) + idx);
        
        float4 grad_in;
        
        grad_in.x = (in_val.x >= -3 && in_val.x <= 3) ? grad_out.x * (1.0f/6.0f) : 0.0f;
        
        grad_in.y = (in_val.y >= -3 && in_val.y <= 3) ? grad_out.y * (1.0f/6.0f) : 0.0f;
        
        grad_in.z = (in_val.z >= -3 && in_val.z <= 3) ? grad_out.z * (1.0f/6.0f) : 0.0f;
        
        // Compute derivative for w component
        grad_in.w = (in_val.w >= -3 && in_val.w <= 3) ? grad_out.w * (1.0f/6.0f) : 0.0f;
        
        reinterpret_cast<float4*>(grad_input)[idx] = grad_in;
    }
}
