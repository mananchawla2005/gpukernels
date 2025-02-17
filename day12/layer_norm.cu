#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float SM[];
    
    float* shared_data = SM;
    
    if (row < rows) {
        float local_sum = 0.0f;
        for (int i = tid; i < cols; i += blockDim.x) {
            shared_data[i] = input[row * cols + i];
            local_sum += shared_data[i];
        }
        
        __shared__ float mean;
        if (tid == 0) mean = 0.0f;
        __syncthreads();
        
        atomicAdd(&mean, local_sum);
        __syncthreads();
        
        if (tid == 0) {
            mean /= cols;
        }
        __syncthreads();
        
        float local_var_sum = 0.0f;
        for (int i = tid; i < cols; i += blockDim.x) {
            float diff = shared_data[i] - mean;
            local_var_sum += diff * diff;
        }
        
        __shared__ float variance;
        if (tid == 0) variance = 0.0f;
        __syncthreads();
        
        atomicAdd(&variance, local_var_sum);
        __syncthreads();
        
        if (tid == 0) {
            variance /= cols;
        }
        __syncthreads();
        
        float stddev = sqrtf(variance + 1e-5f);
        for (int i = tid; i < cols; i += blockDim.x) {
            output[row * cols + i] = (shared_data[i] - mean) / stddev;
        }
    }
}


extern "C" void layer_norm(float* input_h, float* output_h, int rows, int cols) {
    float *input_d, *output_d;
    int size = rows * cols * sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = rows;
    size_t shared_mem_size = (cols) * sizeof(float);
    
    layer_norm_kernel<<<gridSize, blockSize, shared_mem_size>>>(input_d, output_d, rows, cols);
    
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}