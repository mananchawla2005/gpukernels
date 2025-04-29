#include "platform_extensions.hpp"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>


using namespace cute;



int M = 2048, N = 2048;
float *S_d, *D_d;

auto tensor_shape = make_shape(M, N);
auto tensor_shape_trans = make_shape(N, M);
auto gmemLayoutS = make_layout(tensor_shape, GenRowMajor{});
auto gmemLayoutD = make_layout(tensor_shape_trans, GenRowMajor{});



auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});

using SrcTensor = decltype(make_tensor(make_gmem_ptr(S_d), gmemLayoutS));
using DstTensor = decltype(make_tensor(make_gmem_ptr(D_d), gmemLayoutDT));

__global__ void optimized_transpose_kernel(
    SrcTensor tensor_S,
    DstTensor tensor_DT,
    int M_, int N_
) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  if(idx < N_ && idy < M_) {
    tensor_DT(idx, idy) = tensor_S(idy, idx);
  }
}

int main() {
    float *h_input = new float[M * N];
    float *h_output = new float[M * N];
    
    for(int i = 0; i < M * N; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    cudaMalloc(&S_d, M * N * sizeof(float));
    cudaMalloc(&D_d, M * N * sizeof(float));
    
    cudaMemcpy(S_d, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    auto tensor_S = make_tensor(make_gmem_ptr(S_d), gmemLayoutS);
    auto tensor_D = make_tensor(make_gmem_ptr(D_d), gmemLayoutD);
    auto tensor_DT = make_tensor(make_gmem_ptr(D_d), gmemLayoutDT);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    optimized_transpose_kernel<<<gridDim, blockDim>>>(tensor_S, tensor_DT,  M, N);
    
    cudaMemcpy(h_output, D_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Input Matrix (first 4x4):\n");
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%7.1f ", h_input[i * N + j]);
        }
        printf("\n");
    }
    
    printf("\nTransposed Output Matrix (first 4x4):\n");
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%7.1f ", h_output[j*M+i]);  // Always interpret as col-major
        }
        printf("\n");
    }
    
    bool correct = true;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if(h_input[i * N + j] != h_output[i*M+j]) {
                printf("Mismatch at [%d,%d]: input=%.1f, output=%.1f\n",
                       i, j, h_input[i * N + j], h_output[i*M+j]);
                correct = false;
            }
        }
    }
    printf("\nTranspose %s!\n", correct ? "PASSED" : "FAILED");
    
    delete[] h_input;
    delete[] h_output;
    cudaFree(S_d);
    cudaFree(D_d);
    
    return 0;
}