#include <cuda_runtime.h>


__global__ void solution_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; // column index
    const int ty = blockIdx.y * blockDim.y + threadIdx.y; // row index
    
    
    if (tx < m/4 && ty < n) {
        const int idx = ty * (m/4) + tx;
        
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + idx);
        int cx1 = v.x < -3;
        int cx2 = v.x >= 3;
        v.x = 0 * cx1 + (!cx1) * (cx2 * 1 + (!cx2) * ((v.x + 3) / 6));

        // Hard sigmoid for v.y
        int cy1 = v.y < -3;
        int cy2 = v.y >= 3;
        v.y = 0 * cy1 + (!cy1) * (cy2 * 1 + (!cy2) * ((v.y + 3) / 6));

        // Hard sigmoid for v.z
        int cz1 = v.z < -3;
        int cz2 = v.z >= 3;
        v.z = 0 * cz1 + (!cz1) * (cz2 * 1 + (!cz2) * ((v.z + 3) / 6));

        // Hard sigmoid for v.w
        int cw1 = v.w < -3;
        int cw2 = v.w >= 3;
        v.w = 0 * cw1 + (!cw1) * (cw2 * 1 + (!cw2) * ((v.w + 3) / 6));
        
        reinterpret_cast<float4*>(output)[idx] = v;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    
    dim3 blockDim(256, 4); 
    dim3 gridDim((m/4 + blockDim.x - 1) / blockDim.x, 
                 (n + blockDim.y - 1) / blockDim.y);
    
    solution_kernel<<<gridDim, blockDim>>>(input, output, n, m);
}

