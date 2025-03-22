#include <stdio.h>
#include <cuda_runtime.h>

// SWISH activation function: f(x) = x * sigmoid(x)
__global__ void swishKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
    }
}

void swishActivation(float* input_d, float* output_d, int size) {
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    
    swishKernel<<<numBlocks, blockSize>>>(input_d, output_d, size);
}

int main() {
    const int size = 10;
    float input_h[size];
    float output_h[size];
    
    for (int i = 0; i < size; i++) {
        input_h[i] = i - 5.0f; 
    }
    
    float *input_d, *output_d;
    cudaMalloc(&input_d, size * sizeof(float));
    cudaMalloc(&output_d, size * sizeof(float));
    cudaMemcpy(input_d, input_h, size * sizeof(float), cudaMemcpyHostToDevice);
    
    swishActivation(input_d, output_d, size);
    
    cudaMemcpy(output_h, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("SWISH Activation Function Results:\n");
    printf("--------------------------------\n");
    printf("   x    |  SWISH(x)  \n");
    printf("--------------------------------\n");
    
    for (int i = 0; i < size; i++) {
        printf(" %5.2f  |  %8.5f\n", input_h[i], output_h[i]);
    }
    
    cudaFree(input_d);
    cudaFree(output_d);
    
    return 0;
}