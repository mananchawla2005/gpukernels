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
  // tile coords - FIXED: blockIdx.x maps to M dimension, blockIdx.y maps to N dimension
  int bm = blockIdx.x, bn = blockIdx.y, bk = blockIdx.z;
  int row = bm * BLOCK_M + threadIdx.x;  // FIXED: threadIdx.x for row
  int col = bn * BLOCK_N + threadIdx.y;  // FIXED: threadIdx.y for col
  
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
  // FIXED: block dimensions to match thread indexing
  dim3 block(BLOCK_M, BLOCK_N);  // threadIdx.x=BLOCK_M, threadIdx.y=BLOCK_N
  dim3 grid(
    (M + BLOCK_M - 1) / BLOCK_M,  // blockIdx.x covers M dimension
    (N + BLOCK_N - 1) / BLOCK_N,  // blockIdx.y covers N dimension
     split_k                      // blockIdx.z covers K splits
  );
  // C must be zero‐initialized before calling
  splitk_gemm_kernel<<<grid, block, 0>>>(A, B, C, M, N, K, split_k);
}

int main() {
    // problem size and split-K
    const int M = 128, N = 128, K = 128, split_k = 4;
    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(K) * N;
    size_t sizeC = size_t(M) * N;

    // host buffers
    std::vector<float> h_A(sizeA), h_B(sizeB), h_C(sizeC, 0.0f);
    for (size_t i = 0; i < sizeA; ++i) h_A[i] = rand() / float(RAND_MAX);
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = rand() / float(RAND_MAX);

    // device buffers
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    // copy in and zero C
    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeC * sizeof(float));

    // launch
    splitk_gemm(d_A, d_B, d_C, M, N, K, split_k);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // simple check: C[0,0]
    float ref = 0.0f;
    for (int k = 0; k < K; ++k) ref += h_A[k] * h_B[k * N + 0];
    std::cout << "C[0,0] = " << h_C[0] << "  (ref = " << ref << ")\n";

    // ADDED: Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}