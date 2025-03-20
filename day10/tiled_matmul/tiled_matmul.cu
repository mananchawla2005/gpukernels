#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define TILE_WIDTH 32

__global__ void tiled_matmul_kernel(float* M, float* N, float* P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by*TILE_WIDTH+ty;
    int col = bx*TILE_WIDTH+tx;

    float Pval = 0;
    for (int phase = 0; phase < ceil(width/float(TILE_WIDTH)); phase++)
    {
        int m_col = phase * TILE_WIDTH + tx;
        int n_row = phase * TILE_WIDTH + ty;

        if (row < width && m_col < width)
            Mds[ty][tx] = M[row * width + m_col];
        else
            Mds[ty][tx] = 0.0f;
        
        if (n_row < width && col < width)
            Nds[ty][tx] = N[n_row * width + col];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            Pval += Mds[ty][i]*Nds[i][tx];
        }
        __syncthreads();
    }
    if (row < width && col < width)
        P[row * width + col] = Pval;
    
    
}

extern "C" void tiled_matmul(float* M_h, float* N_h, float* P_h, int width) {
    float* M_d, *N_d, *P_d;
    int size = width*width*sizeof(float);
    cudaMalloc((void**)&M_d, size);
    cudaMalloc((void**)&N_d, size);
    cudaMalloc((void**)&P_d, size);
    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(ceil(width/float(TILE_WIDTH)), ceil(width/float(TILE_WIDTH)));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    tiled_matmul_kernel<<<gridSize, blockSize>>>(M_d, N_d, P_d, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}