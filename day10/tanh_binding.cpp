#include <torch/extension.h>

extern "C" void naive_tanh(float* input_h, float* output_h, int n);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "tanh",
        [](torch::Tensor input, torch::Tensor output) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(output.dtype() == torch::kFloat32, "output tensor must be float32");
            
            int n = input.size(0);
            
            float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            
            naive_tanh(input_ptr, output_ptr, n);
        },
        "Performs tanh on the given tensor",
        py::arg("input"),
        py::arg("output")
    );
}