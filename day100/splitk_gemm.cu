#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

#define BLOCK_M 16
#define BLOCK_N 16

// each block computes a TILE_M×TILE_N tile of C over one K-chunk
__global__ void splitk_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float*       C,
    int M, int N, int K,
    int split_k) 
{
  // tile coords - blockIdx.x maps to M dimension, blockIdx.y maps to N dimension
  int bm = blockIdx.x, bn = blockIdx.y, bk = blockIdx.z;
  int row = bm * BLOCK_M + threadIdx.x;  // threadIdx.x for row
  int col = bn * BLOCK_N + threadIdx.y;  //threadIdx.y for col
  
  // compute this block's K‐chunk
  int chunk_size = (K + split_k - 1) / split_k;
  int k0 = bk * chunk_size;
  int k1 = min(k0 + chunk_size, K);

  float acc = 0.0f;
  if (row < M && col < N) {
    for (int k = k0; k < k1; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    // accumulate into C
    atomicAdd(&C[row * N + col], acc);
  }
}

extern "C" void splitk_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K, int split_k) 
{
  // block dimensions to match thread indexing
  dim3 block(BLOCK_M, BLOCK_N);  
  dim3 grid(
    (M + BLOCK_M - 1) / BLOCK_M,  
    (N + BLOCK_N - 1) / BLOCK_N,  
     split_k                      // blockIdx.z covers K splits
  );
  // C must be zero‐initialized before calling
  splitk_gemm_kernel<<<grid, block, 0>>>(A, B, C, M, N, K, split_k);
}

int main() {
    const int M = 128, N = 128, K = 128, split_k = 4;
    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(K) * N;
    size_t sizeC = size_t(M) * N;

    std::vector<float> A_h(sizeA), B_h(sizeB), C_h(sizeC, 0.0f);
    for (size_t i = 0; i < sizeA; ++i) A_h[i] = rand() / float(RAND_MAX);
    for (size_t i = 0; i < sizeB; ++i) B_h[i] = rand() / float(RAND_MAX);

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeA * sizeof(float));
    cudaMalloc(&B_d, sizeB * sizeof(float));
    cudaMalloc(&C_d, sizeC * sizeof(float));

    cudaMemcpy(A_d, A_h.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, sizeC * sizeof(float));

    splitk_gemm(A_d, B_d, C_d, M, N, K, split_k);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h.data(), C_d, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference result on CPU
    std::vector<float> C_ref(sizeC, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A_h[m * K + k] * B_h[k * N + n];
            }
            C_ref[m * N + n] = acc;
        }
    }

    // Verify results
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(C_h[i] - C_ref[i]);
        float rel_diff = diff / std::max(std::abs(C_ref[i]), 1e-6f);
        max_diff = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
    }

    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;
    
    const float tolerance = 1e-5f;
    if (max_rel_diff < tolerance) {
        std::cout << "PASSED: Results match within tolerance" << std::endl;
    } else {
        std::cout << "FAILED: Results differ beyond tolerance" << std::endl;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}