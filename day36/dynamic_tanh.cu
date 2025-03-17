#include <cuda_runtime.h>

__global__ void dynamic_tanh_kernel(float* input, float* output, int n, float* alpha, float* weight, float* bias){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        float scaled_input = alpha[0] * input[idx];
        float ex = expf(scaled_input);
        float emx = expf(-scaled_input);
        float tanh_val = (ex-emx)/(ex+emx);
        output[idx] = tanh_val*weight[idx] + bias[idx];
    }
    
}

extern "C" void dynamic_tanh(float* input_h, float* weight_h, float* output_h, int n, float* alpha_h, float* bias_h){
    float* input_d, *output_d, *weight_d, *alpha_d, *bias_d;
    int size = n*sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMalloc((void**)&weight_d, size);
    cudaMalloc((void**)&bias_d, size);
    cudaMalloc((void**)&alpha_d, sizeof(float));
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_d, weight_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_d, bias_h, size, cudaMemcpyHostToDevice);
    int blockSize = 16*16;
    int gridSize = ceil(n/float(blockSize));
    dynamic_tanh_kernel<<<gridSize, blockSize>>>(input_d, output_d, n, alpha_d, weight_d, bias_d);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(weight_d);
    cudaFree(alpha_d);
    cudaFree(bias_d);
}