#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_profiler_api.h>
__global__ void elu_kernel(const float4* __restrict__ input, float4* __restrict__ output, 
                                    size_t total_new, float alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < total_new) {
        float4 in = __ldg(&input[idx]);
        
        float4 out;
        int cond1 = (in.x > 0.0f);
        int cond2 = (in.y > 0.0f);
        int cond3 = (in.z > 0.0f);
        int cond4 = (in.w > 0.0f);
        out.x = cond1*in.x + (alpha * (__expf(in.x) - 1.0f))*(!cond1);
        out.y = cond2*in.y + (alpha * (__expf(in.y) - 1.0f))*(!cond2);
        out.z = cond3*in.z + (alpha * (__expf(in.z) - 1.0f))*(!cond3);
        out.w = cond4*in.w + (alpha * (__expf(in.w) - 1.0f))*(!cond4);
        
        output[idx] = out;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    int total = n * m;
    int total_new = (total + 3) / 4;
    int blockSize = 256;
    int gridSize = (total_new + blockSize - 1) / blockSize;
    
    elu_kernel<<<gridSize, blockSize>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<float4*>(output),
        total_new, alpha);
        
    cudaDeviceSynchronize();
}

int main() {
    size_t n = 1024;
    size_t m = 1024;
    float alpha = 1.0f;
    size_t total = n * m;

    std::vector<float> input_h(total, 0.5f);
    std::vector<float> output_h(total, 0.0f);

    float *input_d, *output_d;
    cudaMalloc(&input_d, total * sizeof(float));
    cudaMalloc(&output_d, total * sizeof(float));

    cudaMemcpy(input_d, input_h.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    cudaProfilerStart();
    solution(input_d, output_d, n, m, alpha);
    cudaProfilerStop();

    cudaMemcpy(output_h.data(), output_d, total * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First 5 outputs: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output_h[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}