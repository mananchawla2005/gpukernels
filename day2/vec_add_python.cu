#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n){
        C[i]=A[i]+B[i];
    }
}

extern "C" void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    float *A_d, *B_d, *C_d;
    int size = n* sizeof(float);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    // Error Checking
    cudaError_t err = cudaMalloc((void**)&C_d, size);
    if(err!=cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    
    int blocksPerGrid = (n +255 / 256);
    printf("%d", n);
    vecAddKernel<<<blocksPerGrid, 256>>>(A_d, B_d, C_d, n);
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
