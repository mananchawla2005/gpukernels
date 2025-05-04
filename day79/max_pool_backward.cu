#include <cuda_runtime.h>
#include <float.h>  // for FLT_MAX
__global__ void maxpool2d_backward_kernel(const float *__restrict__ input,
                                          const float *__restrict__ grad_output,
                                          int H, int W,
                                          int kernel_size, int stride, int padding, int dilation,
                                          int H_out, int W_out,
                                          float *__restrict__ grad_input)
{
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= H_out || out_x >= W_out)
        return;

    int out_idx = out_y * W_out + out_x;
    float max_val = -FLT_MAX;
    int max_idx = -1;

    // find max element in the pooling window
    for (int m = 0; m < kernel_size; ++m)
    {
        int in_y = out_y * stride + m * dilation - padding;
        for (int n = 0; n < kernel_size; ++n)
        {
            int in_x = out_x * stride + n * dilation - padding;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
            {
                int idx = in_y * W + in_x;
                float v = input[idx];
                if (v > max_val)
                {
                    max_val = v;
                    max_idx = idx;
                }
            }
        }
    }

    // route gradient to the max location
    if (max_idx >= 0)
    {
        atomicAdd(&grad_input[max_idx], grad_output[out_idx]);
    }
}