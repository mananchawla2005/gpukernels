#include <torch/extension.h>

extern "C" void coalased_matmul(float *M_h, float* N_h, float* P_h, int width);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "coalased_matmul",
        [](torch::Tensor M, torch::Tensor N, torch::Tensor P) {
            TORCH_CHECK(M.dtype() == torch::kFloat32, "M tensor must be float32");
            TORCH_CHECK(N.dtype() == torch::kFloat32, "N tensor must be float32");
            TORCH_CHECK(P.dtype() == torch::kFloat32, "P tensor must be float32");
            TORCH_CHECK(M.dim() == 2, "M tensor must be 2D");
            TORCH_CHECK(N.dim() == 2, "N tensor must be 2D");
            TORCH_CHECK(M.size(1) == N.size(0), "Inner matrix dimensions must match");
            
            int width = M.size(1);
            
            float* M_ptr = M.data_ptr<float>();
            float* N_ptr = N.data_ptr<float>();
            float* P_ptr = P.data_ptr<float>();
            
            coalased_matmul(M_ptr, N_ptr, P_ptr, width);
        },
        "Performs coalased matrix multiplication P = M × N",
        py::arg("M"),
        py::arg("N"),
        py::arg("P")
    );
}