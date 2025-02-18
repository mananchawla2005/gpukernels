#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < rows) {
        float local_max = -INFINITY;
        for (int i = tid; i < cols; i += blockDim.x) {
            if(input[row * cols + i]>local_max){
                local_max = input[row*cols+i];
            }
        }
        __shared__ float temp_max[256];  // Assuming max block size of 256
        temp_max[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                temp_max[tid] = max(temp_max[tid], temp_max[tid + stride]);
            }
            // sync before next iteration
            __syncthreads();
        }
        
        float global_max = temp_max[0];
        __syncthreads();

        float local_exp_sum = 0.0f;
        for (int i = tid; i < cols; i += blockDim.x) {
            float exp_val = expf(input[row * cols + i] - global_max);
            local_exp_sum += exp_val;
            output[row * cols + i] = exp_val;  // Store intermediate results
        }
        
        __shared__ float total_sum;
        if (tid == 0) total_sum = 0.0f;
        __syncthreads();
        
        atomicAdd(&total_sum, local_exp_sum);
        __syncthreads();
       
        for (int i = tid; i < cols; i += blockDim.x) {
            output[row * cols + i] /= total_sum;
        }
    }
}


extern "C" void softmax(float* input_h, float* output_h, int rows, int cols) {
    float *input_d, *output_d;
    int size = rows * cols * sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = rows;
    
    softmax_kernel<<<gridSize, blockSize>>>(input_d, output_d, rows, cols);
    
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}