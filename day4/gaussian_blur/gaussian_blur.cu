#include <math.h>
#include <cuda_runtime.h>
#define PI 3.1415926535
__device__ float gaussian(int x, int y, float sigma){
    return exp(-(x*x+y*y)/(2.0f*sigma*sigma)) / (2.0f*PI*sigma*sigma);
}

__global__ void gaussian_blur_kernel(unsigned char* Pin_d, unsigned char* Pout_d, int width, int height, int stride) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int column = blockIdx.x*blockDim.x + threadIdx.x;
    if(row<height && column<width) {
        float sigma = stride/3.0f; // A convention
        int center_elem = stride/2;
        float blur_pixel[3] = {0.0f, 0.0f, 0.0f};
        float weight_sum = 0.0f;

        for (int i = row-center_elem; i <= row+center_elem; i++)
        {
            for (int j = column-center_elem; j <= column+center_elem; j++)
            {
                if(i>=0 && j>=0 && i < height && j < width) {
                    float weight = gaussian(i-row, j-column, sigma);
                    int pixel_idx = (i * width + j) * 3;
                    blur_pixel[0] += Pin_d[pixel_idx] * weight;
                    blur_pixel[1] += Pin_d[pixel_idx+1] * weight;
                    blur_pixel[2] += Pin_d[pixel_idx+2] * weight;
                    weight_sum += weight;
                }
            }
        }
        int out_idx = (row*width+column)*3;
        Pout_d[out_idx] = (unsigned char)(blur_pixel[0] / weight_sum);
        Pout_d[out_idx+1] = (unsigned char)(blur_pixel[1] / weight_sum);
        Pout_d[out_idx+2] = (unsigned char)(blur_pixel[2] / weight_sum);
    }
}

extern "C" void gaussian_blur(unsigned char* Pin_h, unsigned char* Pout_h, int width, int height, int stride){
    unsigned char* Pin_d, *Pout_d;
    const int image_size = width*height*3;
    cudaMalloc((void**)&Pout_d, image_size);
    cudaMalloc((void**)&Pin_d, image_size);
    cudaMemcpy(Pin_d, Pin_h, image_size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width/16.0), ceil(height/16.0));
    gaussian_blur_kernel<<<gridSize, blockSize>>>(Pin_d, Pout_d, width, height, stride);
    cudaMemcpy(Pout_h, Pout_d, image_size, cudaMemcpyDeviceToHost);
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}