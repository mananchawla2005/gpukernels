#include <cuda_runtime.h>
#include <math.h>
#define TILE_WIDTH 16

__global__ void tiled_matmul_corner_turned_kernel(float* M, float* N, float* P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by*blockDim.y+ty;
    int col = bx*blockDim.x+tx;

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
            Nds[tx][ty] = N[col * width + n_row];
        else
            Nds[tx][ty] = 0.0f;
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

extern "C" void tiled_matmul_corner_turned(float* M_h, float* N_h, float* P_h, int width) {
    float* M_d, *N_d, *P_d;
    int size = width*width*sizeof(float);
    cudaMalloc((void**)&M_d, size);
    cudaMalloc((void**)&N_d, size);
    cudaMalloc((void**)&P_d, size);
    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width/16.0), ceil(width/16.0));
    tiled_matmul_corner_turned_kernel<<<gridSize, blockSize>>>(M_d, N_d, P_d, width);
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}