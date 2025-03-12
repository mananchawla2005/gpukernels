import torch
import torch.nn as nn
import gpu_kernels
from transformers import AutoModelForCausalLM, AutoTokenizer

def quantize_weight_custom_double(weight, block_size_outer=128, block_size_inner=32):
    """Quantize using our custom NF4 double quantization implementation"""
    flat_weight = weight.flatten()
    weight_quant, absmax_outer, absmax_inner = gpu_kernels.quantize_nf4_double(
        flat_weight, 
        block_size_outer=block_size_outer,
        block_size_inner=block_size_inner
    )
    return weight_quant, absmax_outer, absmax_inner

def dequantize_nf4_custom_double(weight_quant, absmax_outer, absmax_inner, original_shape, block_size_outer, block_size_inner):
    """Dequantize our custom double-quantized NF4 back to fp32"""
    nf4_codes = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                              0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                              0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
    
    total_elements = original_shape[0] * original_shape[1]
    values = torch.zeros(total_elements, dtype=torch.float32)
    
    # Unpack 4-bit values
    for i in range(weight_quant.shape[0]):
        byte = weight_quant[i].item()
        values[i*2] = nf4_codes[(byte >> 4) & 0xF]
        if i*2 + 1 < total_elements:
            values[i*2 + 1] = nf4_codes[byte & 0xF]
    
    # Apply double scaling
    num_outer_blocks = (total_elements + block_size_outer - 1) // block_size_outer
    num_inner_blocks = (num_outer_blocks + block_size_inner - 1) // block_size_inner
    
    # First apply inner block scaling
    for i in range(num_outer_blocks):
        start_idx = i * block_size_outer
        end_idx = min(start_idx + block_size_outer, total_elements)
        inner_block_idx = i // block_size_inner
        values[start_idx:end_idx] *= absmax_inner[inner_block_idx]
    
    # Then apply outer block scaling with conversion from 8-bit to fp32
    for i in range(num_inner_blocks):
        start_block = i * block_size_inner
        end_block = min(start_block + block_size_inner, num_outer_blocks)
        start_idx = start_block * block_size_outer
        end_idx = min(end_block * block_size_outer, total_elements)
        values[start_idx:end_idx] *= (absmax_outer[i] / 255.0) # CODE8[absmax_outer[i]] for given codebook
    
    return values.reshape(original_shape)

def compare_double_quantization():
    # Load model and tokenizer
    model_id = "C:\\Users\Manan\\.cache\\huggingface\\hub\\models--meta-llama--Llama-3.2-1B-Instruct\\snapshots\\e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading model for custom quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Select a sample layer for comparison
    sample_layer = model.model.layers[0].self_attn.q_proj
    sample_weight = sample_layer.weight.data
    original_shape = sample_weight.shape
    
    print(f"\nTesting on weight matrix shape: {original_shape}")
    print(f"Original weight stats: min={sample_weight.min():.4f}, max={sample_weight.max():.4f}")
    
    
    # Custom double quantization
    print("\nCustom Double-Quantized NF4:")
    block_size_outer = 128
    block_size_inner = 32
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    custom_weight, absmax_outer, absmax_inner = quantize_weight_custom_double(
        sample_weight, 
        block_size_outer=block_size_outer,
        block_size_inner=block_size_inner
    )
    end_event.record()
    torch.cuda.synchronize()
    custom_time = start_event.elapsed_time(end_event)
    
    print(f"Custom quantized packed shape: {custom_weight.shape}")
    print(f"Number of outer absmax values: {len(absmax_outer)}")
    print(f"Number of inner absmax values: {len(absmax_inner)}")
    print(f"Custom quantization time: {custom_time:.3f} ms")
    
    # Dequantize custom for comparison
    dequant_custom = dequantize_nf4_custom_double(
        custom_weight, 
        absmax_outer, 
        absmax_inner, 
        original_shape,
        block_size_outer,
        block_size_inner
    )
    
    # Compare results
    print("\nQuantization Analysis:")
    mse_custom = torch.nn.functional.mse_loss(sample_weight.cpu(), dequant_custom)
    print(f"MSE loss with Custom double-quant: {mse_custom:.6f}")
    
    # Memory analysis
    original_mem = sample_weight.element_size() * sample_weight.nelement()
    quantized_mem = (custom_weight.element_size() * custom_weight.nelement() + 
                    absmax_outer.element_size() * absmax_outer.nelement() +
                    absmax_inner.element_size() * absmax_inner.nelement())
    
    print(f"\nMemory usage for sample layer:")
    print(f"Original: {original_mem/1024:.2f}KB")
    print(f"Quantized (including absmax): {quantized_mem/1024:.2f}KB")
    print(f"Compression ratio: {original_mem/quantized_mem:.2f}x")

if __name__ == "__main__":
    compare_double_quantization()