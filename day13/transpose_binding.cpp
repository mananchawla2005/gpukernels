#include <torch/extension.h>

extern "C" void transpose(float* input_h, float* output_h, int width, int height);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "transpose",
        [](torch::Tensor input, torch::Tensor output) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(output.dtype() == torch::kFloat32, "output tensor must be float32");
            TORCH_CHECK(input.dim() == 2, "input tensor must be 2D");
            TORCH_CHECK(output.dim() == 2, "output tensor must be 2D");
            
            int width = input.size(1);
            int height = input.size(0);
            
            // Check that output dimensions are transposed
            TORCH_CHECK(output.size(0) == width && output.size(1) == height, 
                       "output dimensions must be transposed: expected shape (" 
                       + std::to_string(width) + ", " + std::to_string(height) + ")");

            float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();

            transpose(input_ptr, output_ptr, width, height);
        },
        "Transposes a 2D tensor using CUDA",
        py::arg("input"),
        py::arg("output")
    );
}