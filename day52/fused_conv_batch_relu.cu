#include <math.h>
#include <cuda_runtime.h>

__global__ void fused_conv_batch_relu_kernel(
    const float* __restrict__ input,    
    float* __restrict__ output,          
    const float* __restrict__ weights,      
    const float* __restrict__ bn_weight,    
    const float* __restrict__ bn_bias,     
    const float* __restrict__ bn_mean,     
    const float* __restrict__ bn_var,       
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    float epsilon,
    bool apply_relu)         
{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if (out_row < height && out_col < width) {
        int center = kernel_size / 2;
        
        #pragma unroll
        for (int oc = 0; oc < out_channels; oc++) {
            float conv_result = 0.0f;
            
            #pragma unroll
            for (int ic = 0; ic < in_channels; ic++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        // Calculate input position with stride and padding
                        int in_row = out_row * stride - padding + kh;
                        int in_col = out_col * stride - padding + kw;
                        
                        if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                            int input_idx = ((batch * in_channels + ic) * height + in_row) * width + in_col;
                            int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            conv_result += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
            }
            
            float scale = bn_weight[oc] / sqrtf(bn_var[oc] + epsilon);
            float bias = bn_bias[oc] - bn_mean[oc] * scale;
            float result = fmaxf(0.0f, conv_result * scale + bias);
            
            int output_idx = ((batch * out_channels + oc) * height + out_row) * width + out_col;
            output[output_idx] = result;
        }
    }
}

extern "C" void launch_fused_conv_batch_relu(
    float* input,
    float* output,
    float* weights,
    float* bn_weight,
    float* bn_bias,
    float* bn_mean,
    float* bn_var,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride = 1,
    int padding = 1,
    bool apply_relu = true)
{
    const float epsilon = 1e-5;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        batch_size
    );
    
    fused_conv_batch_relu_kernel<<<gridSize, blockSize>>>(
        input, output, weights, bn_weight, bn_bias,
        bn_mean, bn_var, batch_size,
        in_channels, out_channels, height, width,
        kernel_size, stride, padding, epsilon, apply_relu
    );

    cudaDeviceSynchronize();
}