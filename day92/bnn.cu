#include <stdio.h>
#include <cuda_runtime.h>



__device__ __forceinline__ uint32_t xnor_lop3(uint32_t a, uint32_t b) {
    uint32_t result;
    asm volatile (
        "lop3.b32 %0, %1, %2, %3, 0x96;"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(0xFFFFFFFF)
    );
    return result;
}

__global__ void bnn_dot_kernel(const uint32_t* a, const uint32_t* b, int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t va = a[idx];
        uint32_t vb = b[idx];
        uint32_t xnor = xnor_lop3(va, vb);

        int dot = __popc(xnor) * 2 - 32;  // Convert bit count to dot product
        output[idx] = dot;
    }
}

int main() {
    int N = 1024;
    uint32_t *a, *b;
    int *output;

    cudaMallocManaged(&a, N * sizeof(uint32_t));
    cudaMallocManaged(&b, N * sizeof(uint32_t));
    cudaMallocManaged(&output, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = 0xFFFFFFFF; // All ones
        b[i] = 0xFFFFFFFF; // All ones
    }

    bnn_dot_kernel<<<(N + 255) / 256, 256>>>(a, b, output, N);
    cudaDeviceSynchronize();

    printf("output[0] = %d\n", output[0]);

    cudaFree(a); cudaFree(b); cudaFree(output);
    return 0;
}