#include <torch/extension.h>

extern "C" {
    typedef enum {
        TYPE_FLOAT16,
    } DataType;

    struct quant_state {
        DataType type;
        float* absmax;
        float* code;
        float* offset = {0};  // Match CUDA file initialization
        int blocksize;
    };
        
    struct nf4 {
        uint8_t* weight;
        quant_state quant_state;
    };
}

extern "C" nf4 quantize_nf4(const float *input_h, int n, int block_size_outer, int block_size_inner);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "quantize_nf4",
        [](torch::Tensor input, int block_size_outer, int block_size_inner) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(input.dim() == 1, "input tensor must be 1D");
            
            int n = input.size(0);
            float* input_ptr = input.data_ptr<float>();
            
            // Call CUDA function
            nf4 result = quantize_nf4(input_ptr, n, block_size_outer, block_size_inner);
            
            // Create output tensors
            auto options = torch::TensorOptions().dtype(torch::kUInt8);
            torch::Tensor weight = torch::from_blob(result.weight, {n/2}, options);
            
            auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
            int grid_size = ceil(n/float(block_size_outer));
            torch::Tensor absmax = torch::from_blob(result.quant_state.absmax, {grid_size}, float_options);
            
            // Return as a tuple
            return std::make_tuple(weight, absmax);
        },
        "Quantizes a tensor to NF4 format",
        py::arg("input"),
        py::arg("block_size_outer"),
        py::arg("block_size_inner")
    );
}