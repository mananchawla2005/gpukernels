#include <torch/extension.h>
#include <ATen/ATen.h>

extern "C" void lightning_attention(float* q_h, float* k_h, float* v_h, float* o_h, int n, int d);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "lightning_attention",
        [](torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, c10::optional<torch::Tensor> m) {
            TORCH_CHECK(q.dtype() == torch::kFloat32, "q tensor must be float32");
            TORCH_CHECK(k.dtype() == torch::kFloat32, "k tensor must be float32");
            TORCH_CHECK(v.dtype() == torch::kFloat32, "v tensor must be float32");
            TORCH_CHECK(o.dtype() == torch::kFloat32, "o tensor must be float32");

            TORCH_CHECK(q.dim() == 2, "q tensor must be 2D");
            TORCH_CHECK(k.dim() == 2, "k tensor must be 2D");
            TORCH_CHECK(v.dim() == 2, "v tensor must be 2D");
            TORCH_CHECK(o.dim() == 2, "o tensor must be 2D");

            int n = q.size(0);
            int d = q.size(1);
            TORCH_CHECK(k.size(0) == n && v.size(0) == n && o.size(0) == n,
                        "All input tensors must have the same number of rows");
            TORCH_CHECK(k.size(1) == d && v.size(1) == d && o.size(1) == d,
                        "All input tensors must have the same number of columns");

            float* q_ptr = q.data_ptr<float>();
            float* k_ptr = k.data_ptr<float>();
            float* v_ptr = v.data_ptr<float>();
            float* o_ptr = o.data_ptr<float>();

            lightning_attention(q_ptr, k_ptr, v_ptr, o_ptr, n, d);
        },
        "Computes lightning attention using CUDA",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("o"),
        py::arg("m") = py::none()
    );
}