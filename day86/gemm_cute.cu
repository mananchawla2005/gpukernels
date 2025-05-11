#include "platform_extensions.hpp"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cstdio>

using namespace cute;

// Problem sizes
int M = 512, N = 512, K = 512;
float *A_d, *B_d, *C_d;

// Host/GPU tensor shapes and layouts
auto shapeA = make_shape(M, K);
auto shapeB = make_shape(K, N);
auto shapeC = make_shape(M, N);

auto layoutA = make_layout(shapeA, GenRowMajor{});
auto layoutB = make_layout(shapeB, GenRowMajor{});
auto layoutC = make_layout(shapeC, GenRowMajor{});

using TensorA = decltype(make_tensor(make_gmem_ptr(A_d), layoutA));
using TensorB = decltype(make_tensor(make_gmem_ptr(B_d), layoutB));
using TensorC = decltype(make_tensor(make_gmem_ptr(C_d), layoutC));

__global__ void simple_gemm_kernel(
  TensorA A,
  TensorB B,
  TensorC C,
  int M_, int N_, int K_,
  float alpha, float beta
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < M_ && col < N_) {
    float acc = 0.0f;
    for(int k = 0; k < K_; ++k) {
      acc += A(row, k) * B(k, col);
    }
    C(row, col) = alpha * acc + beta * C(row, col);
  }
}

int main() {
  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);
  float *h_A = new float[M*K];
  float *h_B = new float[K*N];
  float *h_C = new float[M*N];

  for(int i = 0; i < M*K; ++i) h_A[i] = 1.0f;         
  for(int i = 0; i < K*N; ++i) h_B[i] = 2.0f;        
  for(int i = 0; i < M*N; ++i) h_C[i] = 0.5f;        
  cudaMalloc(&A_d, sizeA);
  cudaMalloc(&B_d, sizeB);
  cudaMalloc(&C_d, sizeC);

  cudaMemcpy(A_d, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, h_B, sizeB, cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, h_C, sizeC, cudaMemcpyHostToDevice);

  auto tensorA = make_tensor(make_gmem_ptr(A_d), layoutA);
  auto tensorB = make_tensor(make_gmem_ptr(B_d), layoutB);
  auto tensorC = make_tensor(make_gmem_ptr(C_d), layoutC);

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y);

  // Run GEMM: C = 1.0 * A*B + 0.5 * C
  simple_gemm_kernel<<<grid, block>>>(
    tensorA, tensorB, tensorC,
    M, N, K,
    1.0f, 0.5f
  );
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, C_d, sizeC, cudaMemcpyDeviceToHost);

  // Calculate expected result for verification
  float expected = K * (1.0f * 2.0f) + (0.5f * 0.5f);
  printf("Expected result: %.6f\n", expected);
  
  printf("C[0..3,0..3] =\n");
  for(int i = 0; i < 4; ++i) {
    for(int j = 0; j < 4; ++j) {
      printf("%.6f ", h_C[i*N + j]);
    }
    printf("\n");
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}
