#include <cuda_runtime.h>
#include <math.h>
#define BLOCKSIZE 32

__global__ void coalased_matmul_kernel(float *M, float* N, float* P, int width){
    int row = blockIdx.y*BLOCKSIZE+(threadIdx.x/BLOCKSIZE);
    int col = blockIdx.x*BLOCKSIZE+(threadIdx.x%BLOCKSIZE);
    if(row<width && col<width){
        float rolling_sum = 0;
        for (int i = 0; i < width; i++)
        {
            float M_elem = M[row*width+i];
            float N_elem = N[i*width+col];
            rolling_sum+=M_elem*N_elem;
        }
        P[row*width+col] = rolling_sum;
        
    }
}

extern "C" void coalased_matmul(float *M_h, float* N_h, float* P_h, int width) {
    float *M_d, *N_d, *P_d;
    int matrix_size = width*width*sizeof(float);
    cudaMalloc((void**)&M_d, matrix_size);
    cudaMalloc((void**)&N_d, matrix_size);
    cudaMalloc((void**)&P_d, matrix_size);
    cudaMemcpy(M_d, M_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, matrix_size, cudaMemcpyHostToDevice);
    dim3 blockSize(32 * 32);
    dim3 gridSize(ceil(width/32.0), ceil(width/32.0));
    coalased_matmul_kernel<<<gridSize, blockSize>>>(M_d, N_d, P_d, width);
    cudaMemcpy(P_h, P_d, matrix_size, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}