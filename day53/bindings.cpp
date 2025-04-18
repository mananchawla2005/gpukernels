#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>

extern "C" void stft(const float *waveform,
                     const float *weight_real,
                     const float *weight_imag,
                     float *out_real,
                     float *out_imag,
                     int B,
                     int T,
                     int freq_bins,
                     int filter_length,
                     int stride,
                     int out_width);

extern "C" void adain_snake(const float* input, 
    const float* gamma, 
    const float* beta, 
    const float* mean, 
    const float* var, 
    const float* alpha, 
    float* output, 
    int batch_size, 
    int channels, 
    int width
);

PYBIND11_MODULE(kokoro_kernels, m) {
    m.def(
        "stft",
        [](torch::Tensor waveform, torch::Tensor weight_real, torch::Tensor weight_imag, int stride = 200) {
            TORCH_CHECK(waveform.dtype() == torch::kFloat32, "waveform must be float32");
            TORCH_CHECK(weight_real.dtype() == torch::kFloat32, "weight_real must be float32");
            TORCH_CHECK(weight_imag.dtype() == torch::kFloat32, "weight_imag must be float32");

            auto waveform_contig = waveform.contiguous();
            auto weight_real_contig = weight_real.contiguous();
            auto weight_imag_contig = weight_imag.contiguous();

            // waveform is expected to have shape (B, T)
            int B = waveform_contig.size(0);
            int T = waveform_contig.size(1);

            // weight_real is expected to have shape (freq_bins, 1, filter_length)
            int freq_bins = weight_real_contig.size(0);
            int filter_length = weight_real_contig.size(2);

            // Calculate number of frames; assumes T and stride are provided such that (T - filter_length) is divisible by stride.
            int out_width = (T - filter_length) / stride + 1;

            auto out_real = torch::empty({B, freq_bins, out_width}, waveform.options());
            auto out_imag = torch::empty({B, freq_bins, out_width}, waveform.options());

            stft(waveform_contig.data_ptr<float>(),
                 weight_real_contig.data_ptr<float>(),
                 weight_imag_contig.data_ptr<float>(),
                 out_real.data_ptr<float>(),
                 out_imag.data_ptr<float>(),
                 B,
                 T,
                 freq_bins,
                 filter_length,
                 stride,
                 out_width);

            return py::make_tuple(out_real, out_imag);
        },
        "Fused STFT implementation using CUDA",
        py::arg("waveform"),
        py::arg("weight_real"),
        py::arg("weight_imag"),
        py::arg("stride") = 200
    );
    m.def(
        "adain_snake",
        [](torch::Tensor input, 
           torch::Tensor gamma, 
           torch::Tensor beta,
           torch::Tensor mean,
           torch::Tensor var,
           torch::Tensor alpha) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
            TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma must be float32");
            TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be float32");
            TORCH_CHECK(mean.dtype() == torch::kFloat32, "mean must be float32");
            TORCH_CHECK(var.dtype() == torch::kFloat32, "var must be float32");
            TORCH_CHECK(alpha.dtype() == torch::kFloat32, "alpha must be float32");

            auto input_contig = input.contiguous();
            auto gamma_contig = gamma.contiguous();
            auto beta_contig = beta.contiguous();
            auto mean_contig = mean.contiguous();
            auto var_contig = var.contiguous();
            auto alpha_contig = alpha.contiguous();

            int batch_size = input_contig.size(0);
            int channels = input_contig.size(1);
            int width = input_contig.size(2);

            auto output = torch::empty_like(input_contig);

            adain_snake(
                input_contig.data_ptr<float>(),
                gamma_contig.data_ptr<float>(),
                beta_contig.data_ptr<float>(),
                mean_contig.data_ptr<float>(),
                var_contig.data_ptr<float>(),
                alpha_contig.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                channels,
                width
            );

            return output;
        },
        "Fused AdaIN and Snake activation implementation using CUDA",
        py::arg("input"),
        py::arg("gamma"),
        py::arg("beta"),
        py::arg("mean"),
        py::arg("var"),
        py::arg("alpha")
    );
}