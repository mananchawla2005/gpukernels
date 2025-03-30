#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

// Using the Warp Matrix Multiply-Accumulate (WMMA) API
using namespace nvcuda::wmma;

// Must be multiples of 16 for FP16 operations
#define M 128
#define N 128
#define K 128

__global__ void tensor_core_matmul(half *a, half *b, float *c) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warp_m < M/16 && warp_n < N/16) {
        fill_fragment(c_frag, 0.0f);
        
        for (int i = 0; i < K/16; i++) {
            load_matrix_sync(a_frag, a + warp_m * 16 * K + i * 16, K);
            load_matrix_sync(b_frag, b + i * 16 * N + warp_n * 16, N);
            // The actual Tensor Core operation: C = A × B + C
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        store_matrix_sync(c + warp_m * 16 * N + warp_n * 16, c_frag, N, mem_row_major);
    }
}

void initialize_matrices(half *a, half *b, float *c, int size_a, int size_b, int size_c) {
    for (int i = 0; i < size_a; i++) {
        a[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    for (int i = 0; i < size_b; i++) {
        b[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    for (int i = 0; i < size_c; i++) {
        c[i] = 0.0f;
    }
}

int main() {
    half *a_h = new half[M * K];
    half *b_h = new half[K * N];
    float *c_h = new float[M * N];
    
    initialize_matrices(a_h, b_h, c_h, M * K, K * N, M * N);
    
    half *a_d, *b_d;
    float *c_d;
    cudaMalloc(&a_d, M * K * sizeof(half));
    cudaMalloc(&b_d, K * N * sizeof(half));
    cudaMalloc(&c_d, M * N * sizeof(float));
    
    cudaMemcpy(a_d, a_h, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    dim3 blockDim(128, 4);
    dim3 gridDim((M + (blockDim.x * 16 / 32) - 1) / (blockDim.x * 16 / 32), 
                 (N + blockDim.y * 16 - 1) / (blockDim.y * 16));
    
    tensor_core_matmul<<<gridDim, blockDim>>>(a_d, b_d, c_d);
    
    cudaMemcpy(c_h, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Sample output values:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", c_h[i * N + j]);
        }
        printf("\n");
    }
    
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    
    return 0;
}