#include <torch/extension.h>

extern "C" void layer_norm(float* input_h, float* output_h, int rows, int cols);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "layer_norm",
        [](torch::Tensor input, torch::Tensor output) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(output.dtype() == torch::kFloat32, "output tensor must be float32");
            TORCH_CHECK(input.dim() == 2, "input tensor must be 2D");
            TORCH_CHECK(output.dim() == 2, "output tensor must be 2D");
            TORCH_CHECK(input.sizes() == output.sizes(), "input and output sizes must match");

            int rows = input.size(0);
            int cols = input.size(1);

            float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();

            layer_norm(input_ptr, output_ptr, rows, cols);
        },
        "Applies layer normalization on a 2D tensor",
        py::arg("input"),
        py::arg("output")
    );
}