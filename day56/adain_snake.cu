__global__ void adain_snake_kernel(const float* input, const float* gamma, const float* beta, const float* mean, const float* var, const float* alpha, float* output, int batch_size, int channels, int width)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int total = batch_size*channels*width;
    if(idx>total){
        return;
    }
    int n = idx%width; 
    int f = (idx/width)%channels; 
    int b = (idx/(width*channels));

    float x = input[idx];
    // Instance Normalisation
    float mean_val = mean[b * channels + f];
    float var_val = var[b * channels + f];
    float normalised = (x-mean_val) / sqrt(var_val+1e-5f);

    // AdaIn style transformation
    float gamma_val = gamma[f];
    float beta_val = beta[f];
    float styled = (1.0f+gamma_val)*normalised+beta_val;

    // Snake activation
    // x + (1/a) * (sin(ax))^2
    float alpha_val = alpha[f];
    float sin_term = sin(alpha_val*styled);
    float snake_out = styled+(1.0f/alpha_val)*(sin_term*sin_term);

    output[idx] = snake_out;
}

extern "C" void adain_snake(const float* input, const float* gamma, const float* beta, const float* mean, const float* var, const float* alpha, float* output, int batch_size, int channels, int width) {
    int total = batch_size*channels*width;
    int blockSize = 256;
    int gridSize = ceil(total / float(blockSize));
    adain_snake_kernel<<<gridSize, blockSize>>>(input, gamma, beta, mean, var, alpha, output, batch_size, channels, width);
    cudaDeviceSynchronize();
}