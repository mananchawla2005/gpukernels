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

using SrcTensor = decltype(make_tensor(make_gmem_ptr(S_d), gmemLayoutS));
using DstTensor = decltype(make_tensor(make_gmem_ptr(D_d), gmemLayoutD));

template <int BLOCK_SIZE>
__global__ void tiled_transpose_kernel_shared(
    SrcTensor tensor_S,
    DstTensor tensor_D,
    int M_, int N_)
{
    using b = Int<64>;
    auto block_shape = make_shape(b{}, b{});

    auto smemLayout = make_layout(block_shape, GenRowMajor{});
    auto smemLayoutT = make_layout(block_shape, GenColMajor{});

    extern __shared__ char shared_memory[];
    using CuteArray = array_aligned<float, cosize_v<decltype(smemLayout)>>;
    CuteArray &smem = *reinterpret_cast<CuteArray *>(shared_memory);

    Tensor sS = make_tensor(make_smem_ptr(smem.data()), smemLayout);
    Tensor sD = make_tensor(make_smem_ptr(smem.data()), smemLayoutT);

    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);

    Tensor gS = tiled_tensor_S(make_coord(_, _), blockIdx.y, blockIdx.x);
    Tensor gD = tiled_tensor_D(make_coord(_, _), blockIdx.x, blockIdx.y);

    auto thr_layout = make_layout(make_shape(Int<8>{}, Int<32>{}), GenRowMajor{});

    Tensor tSgS = local_partition(gS, thr_layout, threadIdx.x);
    Tensor tSsS = local_partition(sS, thr_layout, threadIdx.x);
    Tensor tDgD = local_partition(gD, thr_layout, threadIdx.x);
    Tensor tDsD = local_partition(sD, thr_layout, threadIdx.x);

    // GMEM -> SMEM
    copy(tSgS, tSsS);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // SMEM -> GMEM (transposed)
    copy(tDsD, tDgD);
}

int main()
{
    float *h_input = new float[M * N];
    float *h_output = new float[M * N];

    for (int i = 0; i < M * N; i++)
    {
        h_input[i] = static_cast<float>(i);
    }

    cudaMalloc(&S_d, M * N * sizeof(float));
    cudaMalloc(&D_d, M * N * sizeof(float));

    cudaMemcpy(S_d, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    auto tensor_S = make_tensor(make_gmem_ptr(S_d), gmemLayoutS);
    auto tensor_D = make_tensor(make_gmem_ptr(D_d), gmemLayoutD);

    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS = BLOCK_SIZE;
    dim3 blockDim(THREADS, 1);

    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t smem_size = 64 * 64 * sizeof(float);

    tiled_transpose_kernel_shared<BLOCK_SIZE><<<gridDim, blockDim, smem_size>>>(tensor_S, tensor_D, M, N);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
      return -1;
    }
    err = cudaDeviceSynchronize();            // <-- catch any execution errors
    if(err != cudaSuccess) {
        printf("Kernel failed at runtime: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaMemcpy(h_output, D_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input Matrix (first 4x4):\n");
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%7.1f ", h_input[i * N + j]);
        }
        printf("\n");
    }
    printf("\nTransposed Output Matrix (first 4x4):\n");
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%7.1f ", h_output[i * M + j]);
        }
        printf("\n");
    }

    bool correct = true;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (h_input[i * N + j] != h_output[j * M + i])
            {
                printf("Mismatch at [%d,%d]: input=%.1f, output=%.1f\n",
                       i, j, h_input[i * N + j], h_output[j * M + i]);
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