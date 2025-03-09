import triton
import triton.language as tl
import torch
import numpy as np
@triton.jit
def tanh_kernel(input_ptr, output_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask)
    ex = tl.exp(x)
    emx = tl.exp(-x)
    result = (ex - emx) / (ex + emx)
    tl.store(output_ptr + offsets, result, mask=mask)

def naive_tanh(input_h, output_h, n):
    BLOCK = 256
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    tanh_kernel[grid](input_h, output_h, n, BLOCK=BLOCK)

def triton_tanh(x):
    device = x.device
    if device.type == 'cpu':
        x_gpu = x.to('cuda')
    else:
        x_gpu = x
    output = torch.empty_like(x_gpu)

    n = x_gpu.numel()
    naive_tanh(x_gpu, output, n)
    if device.type == 'cpu':
        output = output.to('cpu')
    return output

def demo_tanh():
    size = 10
    np_input = np.random.uniform(-5, 5, size=size)
    input_tensor = torch.tensor(np_input, dtype=torch.float32, device='cuda')
    
    # Compute tanh using our kernel
    triton_output = triton_tanh(input_tensor)
    
    # Compute reference using PyTorch's tanh
    torch_output = torch.tanh(input_tensor)
    
    # Compare results
    print("Random Input and Output Comparison:")
    print(f"{'Input':<10} | {'Triton tanh':<12} | {'PyTorch tanh':<12} | {'Difference':<10}")
    print("-" * 50)
    
    for i in range(size):
        inp = input_tensor[i].item()
        triton_out = triton_output[i].item()
        torch_out = torch_output[i].item()
        diff = abs(triton_out - torch_out)
        print(f"{inp:<10.6f} | {triton_out:<12.6f} | {torch_out:<12.6f} | {diff:<10.6e}")
    
    # Report max error
    max_diff = torch.max(torch.abs(triton_output - torch_output)).item()
    print(f"\nMaximum absolute difference: {max_diff:.6e}")

if __name__ == "__main__":
    demo_tanh()