#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

__global__ void gelu_forward_kernel(float* input_d, float* output_d, int n){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<n){
        float x = input_d[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        
        float tanh_in = sqrt_2_over_pi * (x + coef * x * x * x);
        output_d[idx] = 0.5f * x * (1.0f + tanhf(tanh_in));
    }
}

extern "C" void gelu_forward(float* input_h, float* output_h, int n){
    float *input_d, *output_d;
    int size = n* sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(256);
    dim3 gridSize(ceil(n/256.0));
    gelu_forward_kernel<<<gridSize, blockSize>>>(input_d, output_d, n);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}

__global__ void gelu_backward_kernel(float* grad_output, float* input, float* grad_input, int n){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        float theta = sqrt_2_over_pi*(x+coef*x*x*x);
        float tanh_theta = tanhf(theta);
        float grad = 0.5f * (1.0f+tanh_theta)+0.5f*x*(1-(tanh_theta*tanh_theta))*sqrt_2_over_pi*(1.0f+3.0f*coef*x*x);
        grad_input[idx] = grad*grad_output[idx]; // chain rule
        
    }
}

extern "C" void gelu_backward(float* grad_output_h, float* input_h, float* grad_input_h, int n){
    float *input_d, *grad_input_d, *grad_output_d;
    int size = n* sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&grad_input_d, size);
    cudaMalloc((void**)&grad_output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(grad_output_d, grad_output_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(256);
    dim3 gridSize(ceil(n/256.0));
    gelu_backward_kernel<<<gridSize, blockSize>>>(grad_output_d, input_d, grad_input_d, n);
    cudaMemcpy(grad_input_h, grad_input_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(grad_input_d);
    cudaFree(grad_output_d);
}