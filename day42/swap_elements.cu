#include <stdio.h>
#include <cuda_runtime.h>
__global__ void swapElements(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx % 2 == 0 && idx + 1 < size) {
        int temp = data[idx];
        data[idx] = data[idx + 1];
        data[idx + 1] = temp;
    }
}

int main() {
    const int arraySize = 10;
    int array_h[arraySize]; 
    int* array_d;
    
    printf("Original array: ");
    for (int i = 0; i < arraySize; i++) {
        array_h[i] = i;
        printf("%d ", array_h[i]);
    }
    printf("\n");
    
    cudaMalloc((void**)&array_d, arraySize * sizeof(int));
    
    cudaMemcpy(array_d, array_h, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    
    swapElements<<<blocksPerGrid, threadsPerBlock>>>(array_d, arraySize);
    
    cudaMemcpy(array_h, array_d, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Array after swapping: ");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", array_h[i]);
    }
    printf("\n");
    
    cudaFree(array_d);
    
    return 0;
}