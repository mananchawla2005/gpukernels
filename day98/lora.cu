#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>


// Optimized LoRA forward kernel with shared memory
__global__ void lora_forward_shared_kernel(
    const float* __restrict__ x,      // input [batch_size, in_features]
    const float* __restrict__ W,      // base weights [in_features, out_features]
    const float* __restrict__ A,      // LoRA A [in_features, rank]
    const float* __restrict__ B,      // LoRA B [rank, out_features]
    float* __restrict__ y,            // output [batch_size, out_features]
    const float alpha,                // scaling factor
    const int batch_size,
    const int in_features,
    const int out_features,
    const int rank
) {
    extern __shared__ float shared_mem[];
    
    // Shared memory layout
    float* shared_x = shared_mem;                           // [in_features]
    float* shared_temp = shared_mem + in_features;          // [rank] for intermediate xA
    
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Cooperatively load input row to shared memory
    if (tid < in_features) {
        shared_x[tid] = x[batch_idx * in_features + tid];
    }
    __syncthreads();
    
    if (out_idx >= out_features) return;
    
    // Step 1: Compute base output y = xW
    float base_output = 0.0f;
    for (int i = 0; i < in_features; i++) {
        base_output += shared_x[i] * W[i * out_features + out_idx];
    }
    
    // Step 2: Compute LoRA adaptation
    // First compute xA (store in shared memory for this thread block)
    if (threadIdx.y == 0 && threadIdx.x < rank) {
        float temp_val = 0.0f;
        for (int i = 0; i < in_features; i++) {
            temp_val += shared_x[i] * A[i * rank + threadIdx.x];
        }
        shared_temp[threadIdx.x] = temp_val;
    }
    __syncthreads();
    
    // Then compute (xA)B
    float lora_output = 0.0f;
    for (int r = 0; r < rank; r++) {
        lora_output += shared_temp[r] * B[r * out_features + out_idx];
    }
    
    // Final output: y = xW + α(xA)B
    y[batch_idx * out_features + out_idx] = base_output + alpha * lora_output;
}

// LoRA Layer class
class LoRALayer {
private:
    // Device pointers
    float *W_d, *A_d, *B_d;
    float *x_d, *y_d, *temp_d;
    
    // Dimensions
    int in_features, out_features, rank;
    float alpha;
    
    // cuBLAS handle for optimized matrix operations
    cublasHandle_t cublas_handle;

public:
    LoRALayer(int in_feat, int out_feat, int r, float scaling_factor = 1.0f) 
        : in_features(in_feat), out_features(out_feat), rank(r), alpha(scaling_factor) {
        
        // Allocate device memory
        cudaMalloc(&W_d, in_features * out_features * sizeof(float));
        cudaMalloc(&A_d, in_features * rank * sizeof(float));
        cudaMalloc(&B_d, rank * out_features * sizeof(float));
        
        cublasCreate(&cublas_handle);
        
        // Initialize matrices
        initializeWeights();
    }
    
    ~LoRALayer() {
        cudaFree(W_d);
        cudaFree(A_d);
        cudaFree(B_d);
        if (x_d) cudaFree(x_d);
        if (y_d) cudaFree(y_d);
        if (temp_d) cudaFree(temp_d);
        cublasDestroy(cublas_handle);
    }
    
    void initializeWeights() {
        // Initialize base weights W with random values (simulating pretrained weights)
        std::vector<float> W_h(in_features * out_features);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f / sqrt(in_features));
        
        for (int i = 0; i < W_h.size(); i++) {
            W_h[i] = dist(gen);
        }
        cudaMemcpy(W_d, W_h.data(), W_h.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Initialize A with small random values
        std::vector<float> A_h(in_features * rank);
        std::normal_distribution<float> dist_A(0.0f, 1.0f / sqrt(rank));
        for (int i = 0; i < A_h.size(); i++) {
            A_h[i] = dist_A(gen);
        }
        cudaMemcpy(A_d, A_h.data(), A_h.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Initialize B with zeros (standard LoRA initialization)
        cudaMemset(B_d, 0, rank * out_features * sizeof(float));
    }
    
    void allocateIOBuffers(int max_batch_size) {
        cudaMalloc(&x_d, max_batch_size * in_features * sizeof(float));
        cudaMalloc(&y_d, max_batch_size * out_features * sizeof(float));
        cudaMalloc(&temp_d, max_batch_size * rank * sizeof(float));
    }
    
    void forward(const float* input_h, float* output_h, int batch_size) {
        cudaMemcpy(x_d, input_h, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);
        
        // Configure kernel launch parameters
        dim3 block(1, 16);  // 16 threads per block in y dimension
        dim3 grid(batch_size, (out_features + block.y - 1) / block.y);
        
        // Calculate shared memory size
        size_t shared_mem_size = (in_features + rank) * sizeof(float);
        
        // Launch kernel
        lora_forward_shared_kernel<<<grid, block, shared_mem_size>>>(
            x_d, W_d, A_d, B_d, y_d, alpha,
            batch_size, in_features, out_features, rank
        );
        
        cudaMemcpy(output_h, y_d, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

// Test function
void test_lora_layer() {
    const int batch_size = 4;
    const int in_features = 512;
    const int out_features = 256;
    const int rank = 16;
    const float alpha = 1.0f;
    
    std::cout << "Testing LoRA Layer:" << std::endl;
    std::cout << "Input features: " << in_features << std::endl;
    std::cout << "Output features: " << out_features << std::endl;
    std::cout << "Rank: " << rank << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    
    // Create LoRA layer
    LoRALayer lora(in_features, out_features, rank, alpha);
    lora.allocateIOBuffers(batch_size);
    
    // Create test input
    std::vector<float> input(batch_size * in_features);
    std::vector<float> output(batch_size * out_features);
    
    // Initialize input with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < input.size(); i++) {
        input[i] = dist(gen);
    }
    
    // Test custom kernel
    auto start = std::chrono::high_resolution_clock::now();
    lora.forward(input.data(), output.data(), batch_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_custom = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Custom kernel time: " << duration_custom.count() << " μs" << std::endl;
    
    // Print sample outputs
    std::cout << "\nSample outputs (first batch, first 10 features):" << std::endl;
    std::cout << "Custom: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    test_lora_layer();
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}