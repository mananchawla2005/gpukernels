import torch
import triton
import triton.language as tl
DEVICE = torch.device("cuda:0")


def seeded_dropout(x: torch.Tensor, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['bs']),)
    seeded_dropout_kernel[grid](x, output, n_elements, p, seed, bs=1024)
    return output

@triton.jit
def seeded_dropout_kernel(x, output, n_elements, p # fp32 [0, 1]
                          , seed # int32
                          , bs: tl.constexpr=1024):
    blockId = tl.program_id(0)
    offsets = blockId*bs+tl.arange(0, bs);
    mask = offsets<n_elements
    x = tl.load(x+offsets, mask=mask)
    random = tl.rand(seed, offsets) # uniform distribution 0 to 1 of shape offsets
    x_keep = random>p
    result = tl.where(x_keep, x / (1-p), 0.0)   # accounting for distribution change using /(1-p)
    tl.store(output+offsets, result, mask=mask)

x = torch.randn(size=(8, ), device=DEVICE)
output1 = seeded_dropout(x, 0.2, seed=42)
output2 = seeded_dropout(x, 0.2, seed=1337)
output3 = seeded_dropout(x, 0.2, seed=1)
print(x, output1, output2, output3, sep='\n')
