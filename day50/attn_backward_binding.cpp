#include <torch/extension.h>

extern "C" void self_attn_backward(float* grad_output_h, float* q_h, float* k_h, float* v_h,
                                 float* attn_weights_h,
                                 float* grad_q_h, float* grad_k_h, float* grad_v_h,
                                 int rows, int cols, int d);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "self_attention_backward",
        [](torch::Tensor grad_output, torch::Tensor query, torch::Tensor key, torch::Tensor value,
           torch::Tensor attn_weights, // New parameter
           torch::Tensor grad_query, torch::Tensor grad_key, torch::Tensor grad_value, 
           int d) {
            TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "grad_output must be float32");
            TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
            TORCH_CHECK(key.dtype() == torch::kFloat32, "key must be float32");
            TORCH_CHECK(value.dtype() == torch::kFloat32, "value must be float32");
            TORCH_CHECK(attn_weights.dtype() == torch::kFloat32, "attn_weights must be float32");
            TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D");
            TORCH_CHECK(query.dim() == 2, "query must be 2D");
            TORCH_CHECK(key.dim() == 2, "key must be 2D");
            TORCH_CHECK(value.dim() == 2, "value must be 2D");
            TORCH_CHECK(attn_weights.dim() == 2, "attn_weights must be 2D");
            
            int rows = query.size(0);
            int cols = query.size(1);

            float* grad_output_ptr = grad_output.data_ptr<float>();
            float* query_ptr = query.data_ptr<float>();
            float* key_ptr = key.data_ptr<float>();
            float* value_ptr = value.data_ptr<float>();
            float* attn_weights_ptr = attn_weights.data_ptr<float>();
            float* grad_query_ptr = grad_query.data_ptr<float>();
            float* grad_key_ptr = grad_key.data_ptr<float>();
            float* grad_value_ptr = grad_value.data_ptr<float>();

            self_attn_backward(grad_output_ptr, query_ptr, key_ptr, value_ptr,
                             attn_weights_ptr,
                             grad_query_ptr, grad_key_ptr, grad_value_ptr,
                             rows, cols, d);
        },
        "Computes gradients for self attention",
        py::arg("grad_output"),
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("attn_weights"),
        py::arg("grad_query"),
        py::arg("grad_key"),
        py::arg("grad_value"),
        py::arg("d")
    );
}