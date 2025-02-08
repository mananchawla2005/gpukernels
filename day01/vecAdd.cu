#include <stdio.h>
#include <math.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n){
        C[i]=A[i]+B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n){
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
    
    int blocksPerGrid = (n / 256);
    vecAddKernel<<<blocksPerGrid, 256>>>(A_d, B_d, C_d, n);
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int n = 1024;
    float *A_h = (float*)malloc(n*sizeof(float));
    float *B_h = (float*)malloc(n*sizeof(float));
    float *C_h = (float*)malloc(n*sizeof(float));
    
    for(int i=0; i<n; i++) {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }
    
    vecAdd(A_h, B_h, C_h, n);
    bool correct = true;
    for(int i=0; i<n; i++) {
        printf("%f\n", C_h[i]);
        if(fabs(C_h[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Test %s\n", correct ? "PASSED" : "FAILED");
    
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}