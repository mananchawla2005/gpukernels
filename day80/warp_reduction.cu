#include <cuda_runtime.h>
#include <iostream>

__global__ void row_sum_kernel(const float *matrix, float *output, int num_rows) {
    int row = blockIdx.x;     
    int lane = threadIdx.x;     

    if (row < num_rows && lane < 32) {
        float val = matrix[row * 32 + lane];

        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if (lane == 0) {
            output[row] = val;
        }
    }
}

int main() {
    const int num_rows = 4;
    const int num_cols = 32;

    float h_matrix[num_rows * num_cols];
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            h_matrix[i * num_cols + j] = 1.0f; 
        }
    }

    float *d_matrix, *d_output;
    cudaMalloc(&d_matrix, sizeof(float) * num_rows * num_cols);
    cudaMalloc(&d_output, sizeof(float) * num_rows);

    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice);

    dim3 blockDim(32);      
    dim3 gridDim(num_rows);  
    row_sum_kernel<<<gridDim, blockDim>>>(d_matrix, d_output, num_rows);

    float h_output[num_rows];
    cudaMemcpy(h_output, d_output, sizeof(float) * num_rows, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_rows; ++i) {
        std::cout << "Row " << i << " sum = " << h_output[i] << std::endl;
    }

    cudaFree(d_matrix);
    cudaFree(d_output);
    return 0;
}