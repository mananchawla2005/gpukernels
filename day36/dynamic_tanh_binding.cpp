#include <torch/extension.h>
#include <vector>

extern "C" void dynamic_tanh(float* input_h, float* weight_h, float* output_h, int n, float* alpha_h, float* bias_h);

PYBIND11_MODULE(gpu_kernels, m) {
    // Dynamic tanh binding
    m.def(
        "dynamic_tanh",
        [](torch::Tensor input, torch::Tensor weight, torch::Tensor output, float alpha, torch::Tensor bias) {
            // Input validation
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight tensor must be float32");
            TORCH_CHECK(output.dtype() == torch::kFloat32, "output tensor must be float32");
            TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias tensor must be float32");
            
            TORCH_CHECK(input.size(0) == output.size(0), "input and output tensor must have the same size");
            TORCH_CHECK(input.size(0) == weight.size(0), "input and weight tensor must have the same size");
            TORCH_CHECK(input.size(0) == bias.size(0), "input and bias tensor must have the same size");
            
            int n = input.size(0);
            
            float* input_ptr = input.data_ptr<float>();
            float* weight_ptr = weight.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            float* bias_ptr = bias.data_ptr<float>();
            
            // Call the CUDA function
            dynamic_tanh(input_ptr, weight_ptr, output_ptr, n, &alpha, bias_ptr);
        },
        "Performs dynamic tanh (tanh(alpha*x)*weight + bias) on the given tensor",
        py::arg("input"),
        py::arg("weight"),
        py::arg("output"),
        py::arg("alpha"),
        py::arg("bias")
    );
}