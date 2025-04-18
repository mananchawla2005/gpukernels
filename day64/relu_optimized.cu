#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void solution_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; // column index
    const int ty = blockIdx.y * blockDim.y + threadIdx.y; // row index
    
    if (tx < m/4 && ty < n/4) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = (ty*4 + i) * (m/4) + tx;
            
            float4 v = __ldg(reinterpret_cast<const float4*>(input) + idx);
            
            v.x = fmaxf(v.x, 0.0f);
            v.y = fmaxf(v.y, 0.0f);
            v.z = fmaxf(v.z, 0.0f);
            v.w = fmaxf(v.w, 0.0f);
            
            reinterpret_cast<float4*>(output)[idx] = v;
        }
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    
    dim3 blockDim(32, 8); 
    dim3 gridDim((m/4 + blockDim.x - 1) / blockDim.x, 
                 (n/4 + blockDim.y - 1) / blockDim.y);
    
    solution_kernel<<<gridDim, blockDim>>>(input, output, n, m);
}


void benchmark_relu(size_t n, size_t m, int iterations = 5) {
    n = (n / 4) * 4;
    m = (m / 4) * 4;
    
    size_t size_bytes = n * m * sizeof(float);
    
    float* h_input = new float[n * m];
    float* h_output = new float[n * m];
    
    for (size_t i = 0; i < n * m; i++) {
        h_input[i] = (float)(rand() % 200 - 100) / 10.0f; 
    }
    
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);
    
    cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice);
    
    solution(d_input, d_output, n, m);
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        solution(d_input, d_output, n, m);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double seconds = elapsed.count();
    
    double operations = static_cast<double>(n) * m * iterations;
    double gflops = (operations / seconds) * 1e-9;
    
    std::cout << "Matrix dimensions: " << n << " x " << m << std::endl;
    std::cout << "Total time: " << seconds << " seconds" << std::endl;
    std::cout << "Average time per iteration: " << seconds / iterations * 1000 << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    cudaMemcpy(h_output, d_output, size_bytes, cudaMemcpyDeviceToHost);
    
    int errors = 0;
    for (size_t i = 0; i < n * m; i++) {
        float expected = (h_input[i] > 0) ? h_input[i] : 0.0f;
        if (fabs(h_output[i] - expected) > 1e-5) {
            errors++;
            if (errors < 10) { 
                std::cout << "Error at index " << i << ": expected " << expected 
                          << ", got " << h_output[i] << std::endl;
            }
        }
    }
    
    if (errors > 0) {
        std::cout << "Total errors: " << errors << std::endl;
    } else {
        std::cout << "Verification passed!" << std::endl;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
}

int main(int argc, char** argv) {
    size_t n = 4096;
    size_t m = 4096;
    int iterations = 100;
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    
    n = (n / 4) * 4;
    m = (m / 4) * 4;
    
    benchmark_relu(n, m, iterations);
    
    return 0;
}