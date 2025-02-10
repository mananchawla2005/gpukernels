#include <torch/extension.h>

extern "C" void gelu(float* input_h, float* output_h, int n);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "gelu",
        [](torch::Tensor M, torch::Tensor N) {
            TORCH_CHECK(M.dtype() == torch::kFloat32, "M tensor must be float32");
            TORCH_CHECK(N.dtype() == torch::kFloat32, "N tensor must be float32");
            int n = M.size(1);
            
            float* M_ptr = M.data_ptr<float>();
            float* N_ptr = N.data_ptr<float>();
            
            gelu(M_ptr, N_ptr, n);
        },
        "Performs gelu activation function over a vector",
        py::arg("M"),
        py::arg("N")
    );
}