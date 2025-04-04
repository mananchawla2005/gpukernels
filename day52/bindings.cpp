#include <torch/extension.h>
#include <ATen/ATen.h>

extern "C" void fused_add_relu(float* output_h, float* input1_h, float* input2_h, const int n);
extern "C" void launch_fused_conv_batch_relu(
    float* input,
    float* output,
    float* weights,
    float* bn_weight,
    float* bn_bias,
    float* bn_mean,
    float* bn_var,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size);

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
    m.def(
        "fused_conv_batch_relu",
        [](torch::Tensor input, 
           torch::Tensor weights,
           torch::Tensor bn_weight,
           torch::Tensor bn_bias,
           torch::Tensor bn_mean,
           torch::Tensor bn_var) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights tensor must be float32");
            TORCH_CHECK(bn_weight.dtype() == torch::kFloat32, "bn_weight tensor must be float32");
            TORCH_CHECK(bn_bias.dtype() == torch::kFloat32, "bn_bias tensor must be float32");
            TORCH_CHECK(bn_mean.dtype() == torch::kFloat32, "bn_mean tensor must be float32");
            TORCH_CHECK(bn_var.dtype() == torch::kFloat32, "bn_var tensor must be float32");
            auto input_sizes = input.sizes();
            auto weight_sizes = weights.sizes();
            
            int batch_size = input_sizes[0];
            int in_channels = input_sizes[1];
            int height = input_sizes[2];
            int width = input_sizes[3];
            int out_channels = weight_sizes[0];
            int kernel_size = weight_sizes[2];

            auto output = torch::empty({batch_size, out_channels, height, width}, 
                input.options());

            auto input_contig = input.contiguous();
            auto weights_contig = weights.contiguous();
            auto bn_weight_contig = bn_weight.contiguous();
            auto bn_bias_contig = bn_bias.contiguous();
            auto bn_mean_contig = bn_mean.contiguous();
            auto bn_var_contig = bn_var.contiguous();
            auto output_contig = output.contiguous();

            launch_fused_conv_batch_relu(
                input_contig.data_ptr<float>(),
                output_contig.data_ptr<float>(),
                weights_contig.data_ptr<float>(),
                bn_weight_contig.data_ptr<float>(),
                bn_bias_contig.data_ptr<float>(),
                bn_mean_contig.data_ptr<float>(),
                bn_var_contig.data_ptr<float>(),
                batch_size,
                in_channels,
                out_channels,
                height,
                width,
                kernel_size
            );

            return output;
        },
        "Fused convolution, batch normalization and ReLU operation using CUDA",
        py::arg("input"),
        py::arg("weights"),
        py::arg("bn_weight"),
        py::arg("bn_bias"),
        py::arg("bn_mean"),
        py::arg("bn_var")
    );
}