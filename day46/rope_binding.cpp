#include <torch/extension.h>
#include <ATen/ATen.h>

// Declaration of the RoPE kernel wrapper implemented in rope.cu
extern "C" void rope(float *input_h, float *output_h, int batch_size, int seq_len, int head_dim);

void rope_binding(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output tensor must be float32");
    TORCH_CHECK(input.dim() == 4, "input tensor must be 4D (batch, seq, head, dim)");
    TORCH_CHECK(output.sizes() == input.sizes(), "output tensor shape must match input");

    // Extract dimensions: assuming layout [batch, seq, head, dim]
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int seq_len = sizes[1];
    int num_heads = sizes[2];  // Extract actual head dimension
    int head_dim = sizes[3];   // This is the per-head dimension

    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    rope(input_ptr, output_ptr, batch_size, seq_len, head_dim);
}

PYBIND11_MODULE(gpu_kernels, m) {
    m.def("rope", &rope_binding, "Applies RoPE using a custom CUDA kernel",
          py::arg("input"), py::arg("output"));
}