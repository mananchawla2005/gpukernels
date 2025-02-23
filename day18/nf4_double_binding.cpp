#include <torch/extension.h>

extern "C" {
    typedef enum {
        TYPE_FLOAT16,
    } DataType;

    struct state2 {
        float* absmax;
        float* code;
        int blocksize;
    };

    struct quant_state {
        DataType type;
        uint8_t* absmax;
        float* code;
        float* offset = {0};
        int blocksize;
        state2 state2;
    };
        
    struct nf4 {
        uint8_t* weight;
        quant_state quant_state;
    };
}

extern "C" nf4 quantize_nf4(const float *input_h, int n, int block_size_outer, int block_size_inner);

PYBIND11_MODULE(gpu_kernels, m) {
    m.def(
        "quantize_nf4_double",
        [](torch::Tensor input, int block_size_outer, int block_size_inner) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "input tensor must be float32");
            TORCH_CHECK(input.dim() == 1, "input tensor must be 1D");
            
            int n = input.size(0);
            float* input_ptr = input.data_ptr<float>();
            
            // Call CUDA function
            nf4 result = quantize_nf4(input_ptr, n, block_size_outer, block_size_inner);
            
            // Create output tensors
            auto uint8_options = torch::TensorOptions().dtype(torch::kUInt8);
            auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
            
            // Calculate sizes as integers
            int grid_size_outer = static_cast<int>(ceil(n/static_cast<float>(block_size_outer)));
            int grid_size_inner = static_cast<int>(ceil(grid_size_outer/static_cast<float>(block_size_inner)));
            
            // Quantized weights
            torch::Tensor weight = torch::from_blob(result.weight, {n/2}, uint8_options);
            
            // Outer block absmax values
            torch::Tensor absmax_outer = torch::from_blob(result.quant_state.absmax, 
                {grid_size_inner}, uint8_options);
            
            // Inner block absmax values
            torch::Tensor absmax_inner = torch::from_blob(result.quant_state.state2.absmax, 
                {grid_size_outer/2}, float_options);
            
            // Return as a tuple
            return std::make_tuple(
                weight.clone(),     // Quantized weights
                absmax_outer.clone(), // Outer block absmax values
                absmax_inner.clone()  // Inner block absmax values
            );
        },
        "Quantizes a tensor to NF4 format using double quantization",
        py::arg("input"),
        py::arg("block_size_outer"),
        py::arg("block_size_inner")
    );
}