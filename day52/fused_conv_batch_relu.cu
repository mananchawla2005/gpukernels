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
    float epsilon)         
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if (row < height && col < width) {
        int center = kernel_size / 2;
        
        #pragma unroll
        for (int oc = 0; oc < out_channels; oc++) {
            float conv_result = 0.0f;
            
            #pragma unroll
            for (int ic = 0; ic < in_channels; ic++) {
                for (int kh = -center; kh <= center; kh++) {
                    for (int kw = -center; kw <= center; kw++) {
                        int h = row + kh;
                        int w = col + kw;
                        
                        if (h >= 0 && h < height && w >= 0 && w < width) {
                            int input_idx = ((batch * in_channels + ic) * height + h) * width + w;
                            int weight_idx = ((oc * in_channels + ic) * kernel_size + (kh + center)) * kernel_size + (kw + center);
                            conv_result += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
            }
            
            float scale = bn_weight[oc] / sqrtf(bn_var[oc] + epsilon);
            float bias = bn_bias[oc] - bn_mean[oc] * scale;
            float result = fmaxf(0.0f, conv_result * scale + bias);
            
            int output_idx = ((batch * out_channels + oc) * height + row) * width + col;
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
    int kernel_size)
{
    const float epsilon = 1e-5;

    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    size_t output_size = batch_size * out_channels * height * width * sizeof(float);
    size_t weights_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t bn_size = out_channels * sizeof(float);

    float *d_input, *d_output, *d_weights;
    float *d_bn_weight, *d_bn_bias, *d_bn_mean, *d_bn_var;
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_bn_weight, bn_size);
    cudaMalloc(&d_bn_bias, bn_size);
    cudaMalloc(&d_bn_mean, bn_size);
    cudaMalloc(&d_bn_var, bn_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_weight, bn_weight, bn_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_bias, bn_bias, bn_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_mean, bn_mean, bn_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_var, bn_var, bn_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        batch_size
    );
    
    fused_conv_batch_relu_kernel<<<gridSize, blockSize>>>(
        d_input, d_output, d_weights, d_bn_weight, d_bn_bias,
        d_bn_mean, d_bn_var, batch_size,
        in_channels, out_channels, height, width,
        kernel_size, epsilon
    );

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_bn_weight);
    cudaFree(d_bn_bias);
    cudaFree(d_bn_mean);
    cudaFree(d_bn_var);
}