#include <torch/extension.h>

extern "C" void rgb_to_grayscale(unsigned char * Pout_h, unsigned char * Pin_h, int width, int height);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "rgb_to_grayscale",
        [](torch::Tensor input, torch::Tensor output) {
            TORCH_CHECK(input.dtype() == torch::kUInt8, "Input tensor must be unsigned char");
            TORCH_CHECK(output.dtype() == torch::kUInt8, "Output tensor must be unsigned char");
            TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (height x width x channels)");
            TORCH_CHECK(input.size(2) == 3, "Input tensor must have 3 channels");
            
            int height = input.size(0);
            int width = input.size(1);
            
            unsigned char* input_ptr = input.data_ptr<unsigned char>();
            unsigned char* output_ptr = output.data_ptr<unsigned char>();
            
            rgb_to_grayscale(output_ptr, input_ptr, width, height);
        },
        "Converts RGB image to grayscale",
        py::arg("input"),
        py::arg("output")
    );
}