import triton
import triton.language as tl

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