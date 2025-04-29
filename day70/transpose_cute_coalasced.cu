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

template <int BLOCK_SIZE>
__global__ void tiled_transpose_kernel(
    SrcTensor tensor_S,
    DstTensor tensor_DT,
    int M_, int N_
) {
    using b = Int<BLOCK_SIZE>;
    auto block_shape = make_shape(b{}, b{});       // (b, b)
    
    // Creating tiled views of the tensors
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);   // ([b,b], m/b, n/b)
    Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape); // ([b,b], m/b, n/b)
    
    // Get the tile for this block
    Tensor tile_S = tiled_tensor_S(make_coord(_, _), blockIdx.x, blockIdx.y);
    Tensor tile_DT = tiled_tensor_DT(make_coord(_, _), blockIdx.x, blockIdx.y);
    
    // Defining thread layout for coalesced memory access
    auto thr_layout = make_layout(make_shape(Int<8>{}, Int<32>{}), GenRowMajor{});
    
    // Getting the portion of the tile assigned to this thread
    Tensor thr_tile_S = local_partition(tile_S, thr_layout, threadIdx.x);
    Tensor thr_tile_DT = local_partition(tile_DT, thr_layout, threadIdx.x); 
    
    // Creating register memory and perform the transpose
    Tensor rmem = make_tensor_like(thr_tile_S);
    copy(thr_tile_S, rmem);
    copy(rmem, thr_tile_DT);
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

    constexpr int BLOCK_SIZE = 256;
   constexpr int THREADS = BLOCK_SIZE;
   dim3 blockDim(THREADS, 1);

    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    tiled_transpose_kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(tensor_S, tensor_DT, M, N);
    
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
            printf("%7.1f ", h_output[i*M+j]);  
        }
        printf("\n");
    }
    
    bool correct = true;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if(h_input[i * N + j] != h_output[j * M + i]) {  
                printf("Mismatch at [%d,%d]: input=%.1f, output=%.1f\n",
                       i, j, h_input[i * N + j], h_output[j*M+j]);
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