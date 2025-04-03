#include <torch/extension.h>
#include <ATen/ATen.h>

extern "C" void fused_add_relu(float* output_h, float* input1_h, float* input2_h, const int n);

PYBIND11_MODULE(resnet_kernels, m) {
    m.def(
        "fused_add_relu",
        [](torch::Tensor input1, torch::Tensor input2) {
            TORCH_CHECK(input1.dtype() == torch::kFloat32, "input1 tensor must be float32");
            TORCH_CHECK(input2.dtype() == torch::kFloat32, "input2 tensor must be float32");
            
            TORCH_CHECK(input1.dim() == input2.dim(), "Input tensors must have the same dimensions");
            TORCH_CHECK(input1.sizes() == input2.sizes(), "Input tensors must have the same shape");
            
            auto input1_contig = input1.contiguous();
            auto input2_contig = input2.contiguous();
            
            auto output = torch::empty_like(input1);
            auto output_contig = output.contiguous();
            
            float* input1_ptr = input1_contig.data_ptr<float>();
            float* input2_ptr = input2_contig.data_ptr<float>();
            float* output_ptr = output_contig.data_ptr<float>();
            
            int n = input1.numel();
            
            fused_add_relu(output_ptr, input1_ptr, input2_ptr, n);
            
            return output;
        },
        "Fused add and ReLU operation using CUDA",
        py::arg("input1"),
        py::arg("input2")
    );
}