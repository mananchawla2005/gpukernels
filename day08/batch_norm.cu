#include <cuda_runtime.h>

__global__ void batch_norm_kernel(float* input_d, float* output_d, float* gamma, float* beta, int n, int h, int w, float epsilon){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx>=n*h*w) return;
    int width_idx = idx % w;
    int height_idx = (idx/w) % h;

    float mean = 0.0f;
    for (int i = 0; i < n; i++)
    {
        mean+= input_d[i*h*w+height_idx*w+width_idx];
    }
    mean /= n;

    float variance = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float diff = input_d[i*h*w+height_idx*w+width_idx] - mean;
        variance += diff * diff;
    }
    
    variance /=n;

    int batch_idx = idx/(h*w);
    float x = input_d[idx];
    output_d[idx] = gamma[height_idx*w+width_idx] * (x-mean) / sqrt(variance+epsilon) + beta[height_idx*w+width_idx];
    
    
}
extern "C" void batch_norm(float* input_h, float* output_h, float* gamma_h, float* beta_h, int n, int h, int w, float epsilon){
    float* input_d, *output_d, *gamma_d, *beta_d;
    int total_elements = n*h*w;
    int size = total_elements * sizeof(float);
    int feature_size = h*w * sizeof(float);

    cudaMalloc(&input_d, size);
    cudaMalloc(&output_d, size);
    cudaMalloc(&gamma_d, feature_size);
    cudaMalloc(&beta_d, feature_size);
    int block_size = 256;
    int num_blocks = (total_elements+block_size-1) / block_size;
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_d, gamma_h, feature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, feature_size, cudaMemcpyHostToDevice);
    batch_norm_kernel<<<num_blocks, block_size>>>(input_d, output_d, gamma_d, beta_d, n, h, w, epsilon);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyHostToDevice);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);
}