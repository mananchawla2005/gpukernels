#include <torch/extension.h>

extern "C" void self_attn(float* q_h, float* attn_out_h, int rows, int cols, int d);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "self_attention",
        [](torch::Tensor input, torch::Tensor output, int d) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(input.dim() == 2, "input tensor must be 2D");
            int rows = input.size(0);
            int cols = input.size(1);
            float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            self_attn(input_ptr, output_ptr, rows, cols, d);
        },
        "Computes self attention on input tensor",
        py::arg("input"),
        py::arg("output"),
        py::arg("d")
    );
}