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
}