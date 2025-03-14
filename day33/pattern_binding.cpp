#include <torch/extension.h>

extern "C" void generateImage(unsigned char* output, int width, int height, float time);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "generate_image",
        [](torch::Tensor output, int width, int height, float time) {
            TORCH_CHECK(output.dtype() == torch::kUInt8, "Output tensor must be uint8");
            TORCH_CHECK(output.dim() == 3, "Output tensor must be 3D (height, width, RGBA)");
            TORCH_CHECK(output.size(0) == height, "Output tensor height must match the height argument");
            TORCH_CHECK(output.size(1) == width, "Output tensor width must match the width argument");
            TORCH_CHECK(output.size(2) == 4, "Output tensor must have 4 channels (RGBA)");

            unsigned char* output_ptr = output.data_ptr<unsigned char>();

            generateImage(output_ptr, width, height, time);
        },
        "Generates an image with animated color patterns",
        py::arg("output"),
        py::arg("width"),
        py::arg("height"),
        py::arg("time")
    );
}