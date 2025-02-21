import torch
import torch.nn as nn
import gpu_kernels
import bitsandbytes as bnb
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def quantize_weight_bnb(weight):
    """Quantize using bitsandbytes NF4"""
    return bnb.nn.Params4bit(
        weight.cuda(), 
        requires_grad=False,
        quant_type="nf4"
    ).data

def quantize_weight_custom(weight, block_size=64):
    """Quantize using our custom NF4 implementation"""
    flat_weight = weight.flatten()
    weight_quant, absmax = gpu_kernels.quantize_nf4(
        flat_weight, 
        block_size_outer=block_size,
        block_size_inner=1
    )
    return weight_quant, absmax

def dequantize_nf4_custom(weight_quant, absmax, original_shape):
    """Dequantize our custom NF4 back to fp32 for comparison"""
    # Create lookup table for NF4 codes
    nf4_codes = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                             -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                             0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                             0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
    
    # Extract 4-bit values from uint8
    values = torch.zeros(weight_quant.shape[0] * 2, dtype=torch.float32)
    for i in range(weight_quant.shape[0]):
        byte = weight_quant[i].item()
        values[i*2] = nf4_codes[(byte >> 4) & 0xF]
        values[i*2 + 1] = nf4_codes[byte & 0xF]
    
    # Scale by absmax
    block_size = len(values) // len(absmax)
    for i in range(len(absmax)):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        values[start_idx:end_idx] *= absmax[i]
    
    return values.reshape(original_shape)

def dequantize_nf4_custom(weight_quant, absmax, original_shape):
    """Dequantize our custom NF4 back to fp32 for comparison"""
    # Create lookup table for NF4 codes
    nf4_codes = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                             -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                             0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                             0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
    
    # Calculate total elements needed
    total_elements = original_shape[0] * original_shape[1]
    
    # Extract 4-bit values from uint8
    values = torch.zeros(total_elements, dtype=torch.float32)
    for i in range(weight_quant.shape[0]):  # Iterate over packed bytes
        byte = weight_quant[i].item()
        # Each byte contains two 4-bit values
        values[i*2] = nf4_codes[(byte >> 4) & 0xF]
        if i*2 + 1 < total_elements:  # Check to avoid overflow
            values[i*2 + 1] = nf4_codes[byte & 0xF]
    
    # Scale by absmax per block
    block_size = 64  # Match the block size used in quantization
    num_blocks = (total_elements + block_size - 1) // block_size
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, total_elements)
        values[start_idx:end_idx] *= absmax[i]
    
    return values.reshape(original_shape)

def compare_quantization():
    # Initialize weights with a specific distribution for better testing
    model = SimpleMLP()
    torch.nn.init.normal_(model.fc1.weight, mean=0.0, std=0.02)
    
    # Test quantization on fc1's weight
    weight = model.fc1.weight.data
    original_shape = weight.shape
    
    print(f"Original weight shape: {weight.shape}")
    print(f"Original weight stats: min={weight.min():.4f}, max={weight.max():.4f}")
    
    # Move weight to CUDA for bitsandbytes
    weight_cuda = weight.cuda()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Quantize using bitsandbytes
    print("\nBitsAndBytes NF4 Quantization:")
    start_event.record()
    bnb_weight = quantize_weight_bnb(weight_cuda)
    end_event.record()
    torch.cuda.synchronize()
    bnb_time = start_event.elapsed_time(end_event)
    print(f"BnB quantized shape: {bnb_weight.shape}")
    print(f"BnB stats: min={bnb_weight.min():.4f}, max={bnb_weight.max():.4f}")
    print(f"BnB quantization time: {bnb_time:.3f} ms")

    # Quantize using our implementation
    print("\nCustom NF4 Quantization:")
    custom_weight, absmax = quantize_weight_custom(weight, block_size=64)
    print(f"Custom quantized packed shape: {custom_weight.shape}")
    print(f"Number of absmax values: {len(absmax)}")
    
    # Dequantize our results for comparison
    dequant_custom = dequantize_nf4_custom(custom_weight, absmax, original_shape)
    print(f"Dequantized shape: {dequant_custom.shape}")
    
    # Move everything to CPU for comparison
    weight_cpu = weight
    bnb_weight_cpu = bnb_weight.cpu()
    
    # Compare results
    print("\nQuantization Analysis:")
    mse_bnb = torch.nn.functional.mse_loss(weight_cpu, bnb_weight_cpu)
    mse_custom = torch.nn.functional.mse_loss(weight_cpu, dequant_custom)
    print(f"MSE loss with BnB: {mse_bnb:.6f}")
    print(f"MSE loss with Custom: {mse_custom:.6f}")
    
    # Show some sample values
    print("\nSample Values Comparison (first 5):")
    print("Original:", weight_cpu[0, :5].tolist())
    print("BnB:", bnb_weight_cpu[0, :5].tolist())
    print("Custom:", dequant_custom[0, :5].tolist())
    
    # Memory savings
    original_mem = weight.element_size() * weight.nelement()
    quantized_mem = custom_weight.element_size() * custom_weight.nelement()
    print(f"\nMemory usage:")
    print(f"Original: {original_mem/1024:.2f}KB")
    print(f"Quantized: {quantized_mem/1024:.2f}KB")
    print(f"Compression ratio: {original_mem/quantized_mem:.2f}x")

if __name__ == "__main__":
    compare_quantization()