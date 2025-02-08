#include <torch/extension.h>

extern "C" void gaussian_blur(unsigned char* Pin_h, unsigned char* Pout_h, int width, int height, int stride);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "gaussian_blur",
        [](torch::Tensor input, torch::Tensor output, int stride) {
            TORCH_CHECK(input.dtype() == torch::kUInt8, "Input tensor must be unsigned char");
            TORCH_CHECK(output.dtype() == torch::kUInt8, "Output tensor must be unsigned char");
            TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (height x width x channels)");
            TORCH_CHECK(input.size(2) == 3, "Input tensor must have 3 channels");
            
            int height = input.size(0);
            int width = input.size(1);
            
            unsigned char* input_ptr = input.data_ptr<unsigned char>();
            unsigned char* output_ptr = output.data_ptr<unsigned char>();
            
            gaussian_blur(input_ptr, output_ptr, width, height, stride);
        },
        "Converts RGB image to Gaussian Blurred Image",
        py::arg("input"),
        py::arg("output"),
        py::arg("stride")
    );
}