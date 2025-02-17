__global__ void transpose_kernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

extern "C" void transpose(float* input_h, float* output_h, int width, int height) {
    float *input_d, *output_d;
    int size = width*height*sizeof(float);
    cudaMalloc(&input_d, size);
    cudaMalloc(&output_d, size);
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(32, 32);
    dim3 gridSize(ceil(width/32.0), ceil(height/32.0));
    transpose_kernel<<<gridSize, blockSize>>>(input_d, output_d, width, height);
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}