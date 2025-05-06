#include <cuda_runtime.h>
#include <iostream>

__global__ void row_sum_backward_kernel(float *grad_matrix, const float *grad_output, int num_rows) {
    int row = blockIdx.x;
    int lane = threadIdx.x;

    if (row < num_rows && lane < 32) {
        // Each element in a row contributes equally to the sum
        // So the gradient from the output is distributed equally to all inputs
        float grad = grad_output[row] / 32.0f;
        
        // Write gradient to corresponding matrix element
        grad_matrix[row * 32 + lane] = grad;
    }
}

int main() {
    const int num_rows = 4;
    const int num_cols = 32;

    // Allocate and initialize gradient of output
    float h_grad_output[num_rows];
    for (int i = 0; i < num_rows; ++i) {
        h_grad_output[i] = 1.0f; // Assuming gradient from next layer is 1.0
    }

    // Allocate device memory
    float *d_grad_matrix, *d_grad_output;
    cudaMalloc(&d_grad_matrix, sizeof(float) * num_rows * num_cols);
    cudaMalloc(&d_grad_output, sizeof(float) * num_rows);

    // Copy gradient of output to device
    cudaMemcpy(d_grad_output, h_grad_output, sizeof(float) * num_rows, cudaMemcpyHostToDevice);

    // Launch backward kernel
    dim3 blockDim(32);
    dim3 gridDim(num_rows);
    row_sum_backward_kernel<<<gridDim, blockDim>>>(d_grad_matrix, d_grad_output, num_rows);

    // Allocate and copy results back to host
    float h_grad_matrix[num_rows * num_cols];
    cudaMemcpy(h_grad_matrix, d_grad_matrix, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost);

    // Print some results
    for (int i = 0; i < num_rows; ++i) {
        std::cout << "Gradients for row " << i << ": ";
        for (int j = 0; j < 32; ++j) {
            std::cout << h_grad_matrix[i * num_cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_grad_matrix);
    cudaFree(d_grad_output);
    return 0;
}