#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define CHANNELS 3

__global__
void rgb_to_grayscale_kernel(unsigned char * Pout, unsigned char * Pin, int width, int height){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col<width && row<height) {
        int grayscale_offset = row*width+col;
        int rgb_offset = grayscale_offset * CHANNELS;
        unsigned char r = Pin[rgb_offset];
        unsigned char g = Pin[rgb_offset+1];
        unsigned char b = Pin[rgb_offset+2];

        Pout[grayscale_offset] = 0.21f*r+0.72f*g+0.07f*b;
    }
}

extern "C" void rgb_to_grayscale(unsigned char * Pout_h, unsigned char * Pin_h, int width, int height) {
    unsigned char *Pin_d, *Pout_d;
    const int rgb_size = width * height * CHANNELS;
    const int grayscale_size = width * height;
    cudaMalloc((void**)&Pin_d, rgb_size);
    cudaMalloc((void**)&Pout_d, grayscale_size);
    cudaMemcpy(Pin_d, Pin_h, rgb_size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width/16.0), ceil(height/16.0));
    rgb_to_grayscale_kernel<<<gridSize, blockSize>>>(Pout_d, Pin_d, width, height);
    cudaMemcpy(Pout_h, Pout_d, grayscale_size, cudaMemcpyDeviceToHost);
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}
