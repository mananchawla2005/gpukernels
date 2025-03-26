#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x_cubed = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x_cubed)));
}

__device__ float ge_gelu(float x, float gate) {
    return gate * gelu(x);
}

__global__ void ge_gelu_kernel(float* input_d, float* gate_d, float* output_d, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output_d[idx] = ge_gelu(input_d[idx], gate_d[idx]);
    }
}

int main() {
    const int size = 1024;
    const int bytes = size * sizeof(float);

    float *input_h = (float*)malloc(bytes);
    float *gate_h = (float*)malloc(bytes);
    float *output_h = (float*)malloc(bytes);
    float *input_d, *gate_d, *output_d;

    cudaMalloc(&input_d, bytes);
    cudaMalloc(&gate_d, bytes);
    cudaMalloc(&output_d, bytes);

    for (int i = 0; i < size; i++) {
        input_h[i] = (float)(i - size/2) / (size/4);
        gate_h[i] = (float)i / size;
    }

    cudaMemcpy(input_d, input_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gate_d, gate_h, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    ge_gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, gate_d, output_d, size);

    cudaMemcpy(output_h, output_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(gate_d);
    cudaFree(output_d);
    free(input_h);
    free(gate_h);
    free(output_h);

    return 0;
}