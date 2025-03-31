#include <stdio.h>
#include <cuda_runtime.h>

__global__ void rope_backward_kernel(float *grad_output, float *grad_input, int batch_size, int seq_len, int num_heads, int head_dim, float *cos_cache, float *sin_cache)
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
            
            int paired_idx = current_idx + 1; 
            
            float grad_x = grad_output[current_idx];  
            float grad_y = grad_output[paired_idx];
            
            // Apply rotation
            grad_input[current_idx] = grad_x * cos_val + grad_y * sin_val;     // Even output
            grad_input[paired_idx] = -grad_x * sin_val + grad_y * cos_val;      // Odd output
        }
    }
}

// grad_x = grad_out_x * cos(θ) + grad_out_y * sin(θ)
// grad_y = -grad_out_x * sin(θ) + grad_out_y * cos(θ)