#include <torch/extension.h>

extern "C" void batch_norm(float* input_h, float* output_h, float* gamma_h, float* beta_h, 
                          int n, int h, int w, float epsilon);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "batch_norm",
        [](torch::Tensor input, torch::Tensor output, 
           torch::Tensor gamma, torch::Tensor beta, float epsilon) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(output.dtype() == torch::kFloat32, "output tensor must be float32");
            TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma tensor must be float32");
            TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta tensor must be float32");
            
            int n = input.size(0);  // batch size
            int h = input.size(1);  // height
            int w = input.size(2);  // width
            
            TORCH_CHECK(output.size(0) == n && output.size(1) == h && output.size(2) == w,
                       "output tensor dimensions must match input");
            TORCH_CHECK(gamma.size(0) == h && gamma.size(1) == w,
                       "gamma tensor dimensions must be [height, width]");
            TORCH_CHECK(beta.size(0) == h && beta.size(1) == w,
                       "beta tensor dimensions must be [height, width]");
            
            float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            float* gamma_ptr = gamma.data_ptr<float>();
            float* beta_ptr = beta.data_ptr<float>();
            
            batch_norm(input_ptr, output_ptr, gamma_ptr, beta_ptr, n, h, w, epsilon);
        },
        "Performs batch normalization on input tensor",
        py::arg("input"),
        py::arg("output"),
        py::arg("gamma"),
        py::arg("beta"),
        py::arg("epsilon") = 1e-5
    );
}