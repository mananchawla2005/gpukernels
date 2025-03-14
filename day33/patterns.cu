#include <cuda_runtime.h>

__global__ void generateImageKernel(unsigned char* output, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4; // RGBA format (4 bytes per pixel)
        
        float r = 0.5f + 0.5f * sin(x * 0.01f + time);
        float g = 0.5f + 0.5f * sin(y * 0.01f + time * 0.7f);
        float b = 0.5f + 0.5f * sin((x + y) * 0.01f + time * 1.3f);
        
        output[idx]     = (unsigned char)(r * 255); // R
        output[idx + 1] = (unsigned char)(g * 255); // G
        output[idx + 2] = (unsigned char)(b * 255); // B
        output[idx + 3] = 255;                      // A (fully opaque)
    }
}

extern "C" void generateImage(unsigned char* output, int width, int height, float time) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    unsigned char* d_output;
    cudaMalloc(&d_output, width * height * 4 * sizeof(unsigned char));
    generateImageKernel<<<gridSize, blockSize>>>(d_output, width, height, time);
    cudaMemcpy(output, d_output, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}