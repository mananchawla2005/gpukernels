#include <cuda_runtime.h>
#include <math.h>

__global__ void matmul_kernel(float *M, float* N, float* P, int width){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
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

extern "C" void matmul(float *M_h, float* N_h, float* P_h, int width) {
    float *M_d, *N_d, *P_d;
    int matrix_size = width*width*sizeof(float);
    cudaMalloc((void**)&M_d, matrix_size);
    cudaMalloc((void**)&N_d, matrix_size);
    cudaMalloc((void**)&P_d, matrix_size);
    cudaMemcpy(M_d, M_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, matrix_size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width/16.0), ceil(width/16.0));
    matmul_kernel<<<gridSize, blockSize>>>(M_d, N_d, P_d, width);
    cudaMemcpy(P_h, P_d, matrix_size, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}