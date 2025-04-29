#include <cuda_runtime.h>

__global__ void avgpool2d_kernel(const float *__restrict__ input,
                                 int H, int W,
                                 int kernel_size, int stride, int padding, int dilation,
                                 int H_out, int W_out,
                                 float *__restrict__ output)
{
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= H_out || out_x >= W_out)
        return;

    float sum = 0.0f;
    int count = 0;
    for (int m = 0; m < kernel_size; ++m)
    {
        int in_y = out_y * stride + m * dilation - padding;
        for (int n = 0; n < kernel_size; ++n)
        {
            int in_x = out_x * stride + n * dilation - padding;

            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
            {
                sum += input[in_y * W + in_x];
                count++;
            }
        }
    }
    output[out_y * W_out + out_x] = count > 0 ? sum / count : 0.0f;
}

extern "C" void solution(const float *input,
                         int kernel_size,
                         int stride,
                         int padding,
                         int dilation,
                         float *output,
                         size_t H,
                         size_t W)
{
    int H_out = (int)(((int)H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    int W_out = (int)(((int)W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    dim3 blockSize(16, 16);
    dim3 gridSize((W_out + blockSize.x - 1) / blockSize.x,
              (H_out + blockSize.y - 1) / blockSize.y);

    avgpool2d_kernel<<<gridSize, blockSize>>>(
        input,
        (int)H, (int)W,
        kernel_size, stride, padding, dilation,
        H_out, W_out,
        output);
}
