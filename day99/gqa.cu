#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void group_query_attention_kernel(
    const float* Q,           // [batch, seq_len, num_q_heads, head_dim]
    const float* K,           // [batch, seq_len, num_kv_heads, head_dim]
    const float* V,           // [batch, seq_len, num_kv_heads, head_dim]
    float* O,                 // [batch, seq_len, num_q_heads, head_dim]
    const int batch_size,
    const int seq_len,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    int batch_idx = blockIdx.x;
    int q_head_idx = blockIdx.y;
    int seq_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || q_head_idx >= num_q_heads || seq_idx >= seq_len) {
        return;
    }
    
    // Map query head to corresponding KV head for grouped attention
    int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
    
    extern __shared__ float sdata[];
    float* scores = sdata;
    
    // Compute attention scores (Q * K^T)
    for (int k_seq = 0; k_seq < seq_len; k_seq++) {
        float score = 0.0f;
        
        // Compute Q * K^T for this position
        for (int d = 0; d < head_dim; d++) {
            int q_idx = batch_idx * seq_len * num_q_heads * head_dim + 
                       seq_idx * num_q_heads * head_dim + 
                       q_head_idx * head_dim + d;
            
            int k_idx = batch_idx * seq_len * num_kv_heads * head_dim + 
                       k_seq * num_kv_heads * head_dim + 
                       kv_head_idx * head_dim + d;
            
            score += Q[q_idx] * K[k_idx];
        }
        
        scores[k_seq] = score * scale;
    }
    
    __syncthreads();
    
    // Apply softmax normalization
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_score = fmaxf(max_score, scores[i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    
    for (int i = 0; i < seq_len; i++) {
        scores[i] /= sum_exp;
    }
    
    __syncthreads();
    
    // Compute weighted sum with values
    for (int d = 0; d < head_dim; d++) {
        float output_val = 0.0f;
        
        for (int v_seq = 0; v_seq < seq_len; v_seq++) {
            int v_idx = batch_idx * seq_len * num_kv_heads * head_dim + 
                       v_seq * num_kv_heads * head_dim + 
                       kv_head_idx * head_dim + d;
            
            output_val += scores[v_seq] * V[v_idx];
        }
        
        int o_idx = batch_idx * seq_len * num_q_heads * head_dim + 
                   seq_idx * num_q_heads * head_dim + 
                   q_head_idx * head_dim + d;
        
        O[o_idx] = output_val;
    }
}

void launch_group_query_attention(
    const float* d_Q,
    const float* d_K, 
    const float* d_V,
    float* d_O,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    dim3 grid(batch_size, num_q_heads);
    dim3 block(seq_len);
    size_t shared_mem_size = seq_len * sizeof(float);
    
    group_query_attention_kernel<<<grid, block, shared_mem_size>>>(
        d_Q, d_K, d_V, d_O,
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
}

int main() {
    // Model configuration
    const int batch_size = 2;
    const int seq_len = 128;
    const int num_q_heads = 8;
    const int num_kv_heads = 2;  // Fewer KV heads for grouped attention
    const int head_dim = 64;
    
    std::cout << "Group Query Attention CUDA Implementation\n";
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Sequence length: " << seq_len << std::endl;
    std::cout << "Query heads: " << num_q_heads << std::endl;
    std::cout << "Key-Value heads: " << num_kv_heads << std::endl;
    std::cout << "Head dimension: " << head_dim << std::endl;
    
    // Calculate tensor sizes
    size_t q_size = batch_size * seq_len * num_q_heads * head_dim;
    size_t kv_size = batch_size * seq_len * num_kv_heads * head_dim;
    size_t o_size = batch_size * seq_len * num_q_heads * head_dim;
    
    // Allocate host memory
    std::vector<float> Q_h(q_size);
    std::vector<float> K_h(kv_size);
    std::vector<float> V_h(kv_size);
    std::vector<float> O_h(o_size);
    
    // Initialize with random values
    for (size_t i = 0; i < q_size; i++) {
        Q_h[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (size_t i = 0; i < kv_size; i++) {
        K_h[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        V_h[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    
    // Allocate device memory
    float *Q_d, *K_d, *V_d, *O_d;
    cudaMalloc(&Q_d, q_size * sizeof(float));
    cudaMalloc(&K_d, kv_size * sizeof(float));
    cudaMalloc(&V_d, kv_size * sizeof(float));
    cudaMalloc(&O_d, o_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(Q_d, Q_h.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Time the kernel execution
    cudaEventRecord(start);
    
    // Launch the GQA kernel
    launch_group_query_attention(Q_d, K_d, V_d, O_d, 
                                batch_size, seq_len, num_q_heads, num_kv_heads, head_dim);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(O_h.data(), O_d, o_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    
    // Print first few output values for verification
    std::cout << "First 10 output values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << O_h[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(O_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "Group Query Attention completed successfully!" << std::endl;
    
    return 0;
}