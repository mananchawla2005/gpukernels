#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

int calculate_tile_width(cudaDeviceProp& prop, int width) {
    int max_threads = prop.maxThreadsPerBlock;
    int max_dim = (int)sqrt(max_threads);
    size_t shared_mem = prop.sharedMemPerBlock * 0.9; // 90% of available
    int mem_limited_tile = (int)sqrt(shared_mem / (2 * sizeof(float)));
    int size_limited_tile = width;
    int tile_width = max_dim;
    
    if (mem_limited_tile < tile_width) tile_width = mem_limited_tile;
    if (width < tile_width) tile_width = width;
    
    // Round down to nearest power of 2 for better performance
    if (tile_width > 0) {
        tile_width = (int)pow(2, floor(log2(tile_width)));
    } else {
        tile_width = 16; // fallback to a safe default
    }
    
    // Ensure tile_width is at least 1 and not larger than width
    tile_width = max(1, min(tile_width, width));
    
    printf("Using Tile Width: %d\n", tile_width);
    
    return tile_width;
}

__global__ void dynamic_tiled_matmul_kernel(float* M, float* N, float* P, int width, int TILE_WIDTH, unsigned Mds_sz, unsigned Nds_sz) {
    extern __shared__ float Mds_Nds[];
    unsigned Mds_elements = Mds_sz / sizeof(float);
    float *Mds = (float*) Mds_Nds;
    float *Nds = (float *) Mds_Nds + Mds_elements;
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
            Mds[ty*TILE_WIDTH+tx] = M[row * width + m_col];
        else
            Mds[ty*TILE_WIDTH+tx] = 0.0f;
        
        if (n_row < width && col < width)
            Nds[ty*TILE_WIDTH+tx] = N[n_row * width + col];
        else
            Nds[ty*TILE_WIDTH+tx] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            Pval += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
        }
        __syncthreads();
    }
    if (row < width && col < width)
        P[row * width + col] = Pval;
    
    
}
extern "C" void dynamic_tiled_matmul(float* M_h, float* N_h, float* P_h, int width) {
    float* M_d, *N_d, *P_d;
    int size = width*width*sizeof(float);
    cudaDeviceProp devProp;
    cudaError_t err = cudaGetDeviceProperties(&devProp, 0);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }
    int tile_width = calculate_tile_width(devProp, width);
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);
    cudaMalloc((void**)&M_d, size);
    cudaMalloc((void**)&N_d, size);
    cudaMalloc((void**)&P_d, size);
    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(tile_width, tile_width);
    dim3 gridSize(ceil(width/float(tile_width)), ceil(width/float(tile_width)));
    dynamic_tiled_matmul_kernel<<<gridSize, blockSize, shared_mem_size>>>(M_d, N_d, P_d, width, tile_width, shared_mem_size/2, shared_mem_size/2);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(M_d);
        cudaFree(N_d);
        cudaFree(P_d);
        return;
    }
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}