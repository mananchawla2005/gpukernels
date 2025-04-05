#include <cuda_runtime.h>
// Parameters:
// waveform: (B, T)
// weight_real, weight_imag: (freq_bins, filter_length)
// out_real, out_imag: (B, freq_bins, out_width)
// stride: hop length
// T: length of waveform (after any padding)
// filter_length: kernel size, typically n_fft
// out_width: number of frames = (T - filter_length) / stride + 1

__global__ void stft_kernel(const float *__restrict__ waveform,
                            const float *__restrict__ weight_real,
                            const float *__restrict__ weight_imag,
                            float *__restrict__ out_real,
                            float *__restrict__ out_imag,
                            int B,
                            int T,
                            int freq_bins,
                            int filter_length,
                            int stride,
                            int out_width)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int total = B*freq_bins*out_width;
    if(idx>total)
        return;
    int n = idx%out_width; // frame index (time)
    int f = (idx/out_width)%freq_bins; // freq bin
    int b = (idx/(out_width*freq_bins));

    float sum_real = 0.0f;
    float sum_imag = 0.0f;
    int input_offset = b*T+n*stride;
    int weight_offset = f*filter_length;

    for (int k = 0; k < filter_length; k++)
    {
        float x = waveform[input_offset+k];
        sum_real += x*weight_real[weight_offset+k];
        sum_imag += x*weight_imag[weight_offset+k];
    }

    out_real[idx] = sum_real;
    out_imag[idx] = sum_imag;
    
}

extern "C" void stft(const float *waveform, const float *weight_real, const float *weight_imag, float *out_real, float *out_imag, int B, int T, int freq_bins, int filter_length, int stride, int out_width)
{
    int total = B * freq_bins * out_width;
    int blockSize = 256;
    int gridSize = ceil(total / float(blockSize));
    stft_kernel<<<gridSize, blockSize>>>(waveform, weight_real, weight_imag, out_real, out_imag, B, T, freq_bins, filter_length, stride, out_width);

    cudaDeviceSynchronize();
}