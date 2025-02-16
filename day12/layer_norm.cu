#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(float* input, float* output, int rows, int cols){
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows) {
        extern __shared__ float SM[];
        float* shared_data = SM;

        for (size_t i = threadIdx.y; i < cols; i+=blockDim.y)
        {
            shared_data[i] = input[row*cols+i];
        }
        
        __syncthreads();

        float mean = 0.0f;
        for (size_t i = threadIdx.y; i < cols; i+=blockDim.y)
        {
            mean+=shared_data[i];
        }
        mean /= cols;

        float variance = 0.0f;
        for (int i = threadIdx.y; i < cols; i+=blockDim.y)
        {
            float diff = shared_data[i] - mean;
            variance += diff*diff;
        }
        variance/=cols;

        float stddev = sqrtf(variance + 1e-5);

        for (size_t i = threadIdx.y; i < cols; i+=blockDim.y)
        {
            output[row*cols+i] = (shared_data[i]-mean) / stddev; 
        }
        
        
        
    }
    
}   


extern "C" void layer_norm(float* input_h, float* output_h, int rows, int cols) {
    float *input_d, *output_d;
    int size = rows*cols*sizeof(float);
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void**)&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = ceil(rows/float(blockSize));
    size_t shared_mem_size = cols * sizeof(float);
    layer_norm_kernel<<<gridSize, blockSize, shared_mem_size>>>(input_d, output_d, rows, cols);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);

}