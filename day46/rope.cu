#include <stdio.h>
#include <cuda_runtime.h>

__global__ void rope_kernel(float *input, float *output, int batch_size, int seq_len, int num_heads, int head_dim, float *cos_cache, float *sin_cache)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * num_heads * head_dim;
    if (idx < total)
    {
        // Layout: [batch, seq, head, dim]
        int dim_idx = idx % head_dim;
        int head_idx = (idx / head_dim) % num_heads;
        int seq_idx = (idx / (head_dim * num_heads)) % seq_len;
        int batch_idx = idx / (head_dim * num_heads * seq_len);

        // Only process even dimensions (each thread handles a pair)
        if (dim_idx % 2 == 0)
        {
            int pair_dim = dim_idx / 2;
            float cos_val = cos_cache[seq_idx * (head_dim / 2) + pair_dim];
            float sin_val = sin_cache[seq_idx * (head_dim / 2) + pair_dim];
            
            // Calculate indices for the current dimension and its pair
            int current_idx = batch_idx * (seq_len * num_heads * head_dim) + 
                             seq_idx * (num_heads * head_dim) + 
                             head_idx * head_dim + 
                             dim_idx;
            
            int paired_idx = current_idx + 1; // The paired index is always the next one
            
            float x = input[current_idx];    // Even dimension value
            float y = input[paired_idx];     // Odd dimension value
            
            // Apply rotation
            output[current_idx] = x * cos_val - y * sin_val;     // Even output
            output[paired_idx] = x * sin_val + y * cos_val;      // Odd output
        }
    }
}

extern "C" void rope(float *input_h, float *output_h, int batch_size, int seq_len, int head_dim)
{
    int num_heads = 1;
    int total_elements = batch_size * seq_len * num_heads * head_dim;
    size_t size = total_elements * sizeof(float);
    float *input_d, *output_d, *cos_cache_d, *sin_cache_d;
    cudaMalloc(&input_d, size);
    cudaMalloc(&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    int cache_size = seq_len * (head_dim / 2);
    float *cos_cache_h = new float[cache_size];
    float *sin_cache_h = new float[cache_size];
    float *theta = new float[head_dim / 2];
    for (int i = 0; i < head_dim / 2; i++)
    {
        float exponent = (float)(2 * i) / head_dim;
        theta[i] = 1.0f / powf(10000.0f, exponent);
    }
    for (int pos = 0; pos < seq_len; pos++)
    {
        for (int i = 0; i < head_dim / 2; i++)
        {
            float angle = pos * theta[i];
            cos_cache_h[pos * (head_dim / 2) + i] = cosf(angle);
            sin_cache_h[pos * (head_dim / 2) + i] = sinf(angle);
        }
    }

    cudaMalloc((void **)&cos_cache_d, cache_size * sizeof(float));
    cudaMalloc((void **)&sin_cache_d, cache_size * sizeof(float));
    cudaMemcpy(cos_cache_d, cos_cache_h, cache_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sin_cache_d, sin_cache_h, cache_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ceil(total_elements / float(blockSize));
    rope_kernel<<<gridSize, blockSize>>>(input_d, output_d, batch_size, seq_len, num_heads, head_dim, cos_cache_d, sin_cache_d);
    cudaDeviceSynchronize();
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(cos_cache_d);
    cudaFree(sin_cache_d);
    delete[] cos_cache_h;
    delete[] sin_cache_h;
    delete[] theta;
}
