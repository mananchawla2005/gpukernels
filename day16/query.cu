#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int dev_count;
    cudaError_t error = cudaGetDeviceCount(&dev_count);
    
    if (error != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("====================================\n");
    printf("CUDA Device Query\n");
    printf("====================================\n");

    for(int i = 0; i < dev_count; i++) {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, i);
        
        printf("\nDevice %d: \"%s\"\n", i, dev_prop.name);
        printf("------------------------------------\n");
        
        printf("CUDA Capability Major/Minor version number: %d.%d\n", dev_prop.major, dev_prop.minor);
        
        printf("Total Global Memory: %.2f GB\n", dev_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Memory Clock Rate: %.0f MHz\n", dev_prop.memoryClockRate * 1e-3f);
        printf("Memory Bus Width: %d-bit\n", dev_prop.memoryBusWidth);
        printf("L2 Cache Size: %d KB\n", dev_prop.l2CacheSize / 1024);
        
        printf("\nThread/Block Specifications:\n");
        printf("Max Threads per Block: %d\n", dev_prop.maxThreadsPerBlock);
        printf("Max Thread Dimensions: (%d, %d, %d)\n", 
               dev_prop.maxThreadsDim[0],
               dev_prop.maxThreadsDim[1],
               dev_prop.maxThreadsDim[2]);
        printf("Max Grid Dimensions: (%d, %d, %d)\n",
               dev_prop.maxGridSize[0],
               dev_prop.maxGridSize[1],
               dev_prop.maxGridSize[2]);
        
        printf("\nHardware Architecture:\n");
        printf("Number of SMs: %d\n", dev_prop.multiProcessorCount);
        printf("GPU Clock Rate: %.0f MHz\n", dev_prop.clockRate * 1e-3f);
        printf("Warp Size: %d\n", dev_prop.warpSize);
        
        printf("\nFeature Support:\n");
        printf("Concurrent Kernels: %s\n", dev_prop.concurrentKernels ? "Yes" : "No");
        printf("Compute Mode: %d\n", dev_prop.computeMode);
        printf("ECC Enabled: %s\n", dev_prop.ECCEnabled ? "Yes" : "No");
        printf("Unified Addressing: %s\n", dev_prop.unifiedAddressing ? "Yes" : "No");
        
        printf("====================================\n");
    }

    return 0;
}