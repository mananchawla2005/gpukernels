#include <torch/extension.h>

extern "C" void vecAdd(float* A, float* B, float* C, int n);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "vec_add",
        [](torch::Tensor A, torch::Tensor B, torch::Tensor C) {
            int n = A.size(0);
            float* A_ptr = A.data_ptr<float>();
            float* B_ptr = B.data_ptr<float>();
            float* C_ptr = C.data_ptr<float>();
            vecAdd(A_ptr, B_ptr, C_ptr, n);
        },
        "Launches the vector addition kernel",
        py::arg("A"),
        py::arg("B"),
        py::arg("C")
    );
}
