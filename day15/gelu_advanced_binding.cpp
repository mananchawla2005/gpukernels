#include <torch/extension.h>

extern "C" void gelu_forward(float* input_h, float* output_h, int n);
extern "C" void gelu_backward(float* grad_output_h, float* input_h, float* grad_input_h, int n);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "gelu_forward",
        [](torch::Tensor input, torch::Tensor output) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(output.dtype() == torch::kFloat32, "output tensor must be float32");
            TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
            TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous");
            int n = input.numel();
            
            float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            
            gelu_forward(input_ptr, output_ptr, n);
            
            return output;
        },
        "Performs GELU activation function forward pass",
        py::arg("input"),
        py::arg("output")
    );

    m.def(
        "gelu_backward",
        [](torch::Tensor grad_output, torch::Tensor input, torch::Tensor grad_input) {
            TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "grad_output tensor must be float32");
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(grad_input.dtype() == torch::kFloat32, "grad_input tensor must be float32");
            TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor must be contiguous");
            TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
            TORCH_CHECK(grad_input.is_contiguous(), "grad_input tensor must be contiguous");
            int n = input.numel();
            
            float* grad_output_ptr = grad_output.data_ptr<float>();
            float* input_ptr = input.data_ptr<float>();
            float* grad_input_ptr = grad_input.data_ptr<float>();
            
            gelu_backward(grad_output_ptr, input_ptr, grad_input_ptr, n);
            
            return grad_input;
        },
        "Performs GELU activation function backward pass",
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("grad_input")
    );
}