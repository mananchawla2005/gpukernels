#include <cuda_fp16.h>
#include <cuda.h>
#include <stdio.h>

const int TILE_M = 128, TILE_N = 128, TILE_K = 32;
const int WARPS_PER_ROW = TILE_N / 16;  // assuming 16×16 per-warp

template<typename T>
__global__ void async_gemm_prefetch(const T* __restrict__ A,
                                    const T* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K,
                                    int lda, int ldb, int ldc)
{
    // shared-memory buffer for a tile of A and B
    extern __shared__ T shmem[];
    T* shA = shmem;
    T* shB = shmem + TILE_M * TILE_K;

    // thread / warp indices
    int warpId      = threadIdx.x / 32;
    int laneId      = threadIdx.x % 32;
    int warpRow     = warpId / WARPS_PER_ROW;
    int warpCol     = warpId % WARPS_PER_ROW;

    // Compute starting pointers for this block
    int blockRow = blockIdx.y * TILE_M;
    int blockCol = blockIdx.x * TILE_N;

    // each thread’s fragment accumulator (16×16 warp-tile)
    float acc[16];
    #pragma unroll
    for(int i=0; i<16; ++i) acc[i] = 0.0f;

    // Loop over K in chunks of TILE_K
    for(int k0 = 0; k0 < K; k0 += TILE_K) {
        // 1) Issue asynchronous loads of A and B tiles into shared memory
        int offsetA = (blockRow + warpRow*16)*lda + (k0 + laneId);
        int offsetB = (blockCol + warpCol*16)*ldb + (k0 + laneId);

        // Each thread copies one element (vectorize as needed)
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.cg.shared.global [%3], [%4], 16;\n"
            :
            : "l"(shA  + warpRow*16* TILE_K + laneId), 
              "l"(A     + offsetA),                     
              "n"(16),  // Fixed size to 16 bytes
              "l"(shB  + warpCol*16* TILE_K + laneId),  
              "l"(B     + offsetB)
        );

        // Tell the hardware to start these copies in a batch of 1
        asm volatile("cp.async.commit_group;");

        // 2) Wait until that group is done before using the tile
        asm volatile("cp.async.wait_group 0;");

        // 3) Synchronize all threads in the block (to safely read shmem)
        __syncthreads();

        // 4) Do the local warp-level GEMM on the just-loaded tile
        //    (You could use mma.sync here, but keeping it simple)
        for(int kk=0; kk < TILE_K; ++kk) {
            T a_val = shA[warpRow*16* TILE_K + kk*16 + (laneId/16)];
            T b_val = shB[warpCol*16 + kk*16 + (laneId%16)];
            #pragma unroll
            for(int i=0; i<16; ++i)
                acc[i] += float(a_val) * float(b_val);
        }

        __syncthreads();  // for next iteration’s use of shared
    }

    // 5) Write back your accumulated 16×16 warp-tile to C
    int cRow = blockRow + warpRow*16 + (laneId/16);
    int cCol = blockCol + warpCol*16 + (laneId%16);
    C[cRow*ldc + cCol] = acc[ (laneId/16)*16 + (laneId%16) ];
}


// Host launch
int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    size_t bytes_A = M * K * sizeof(__half);
    size_t bytes_B = K * N * sizeof(__half);
    size_t bytes_C = M * N * sizeof(float);

    // Allocate device memory
    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // Initialize input matrices with random values
    __half *h_A = (__half*)malloc(bytes_A);
    __half *h_B = (__half*)malloc(bytes_B);
    
    for(int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(0.1f * (rand() % 10));
    }
    for(int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(0.1f * (rand() % 10));
    }

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(128);  // 4 warps per block
    dim3 grid((N+TILE_N-1)/TILE_N, (M+TILE_M-1)/TILE_M);
    size_t shared_bytes = (TILE_M*TILE_K + TILE_K*TILE_N) * sizeof(__half);
    
    async_gemm_prefetch<__half><<<grid, block, shared_bytes>>>(
        d_A, d_B, d_C, M, N, K, lda, ldb, ldc
    );
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);

    return 0;
}