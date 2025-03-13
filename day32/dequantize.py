from triton import jit
import triton
import triton.language as tl
import torch

@triton.jit
def _your_dequantize_nf4_kernel(
    output_ptr,
    weight_quant_ptr,   
    absmax_outer_ptr,
    absmax_inner_ptr,
    codebook_outer_ptr,
    codebook_inner_ptr,
    n_elements,
    block_size_outer,
    block_size_inner,
    BLOCK_SIZE: tl.constexpr
):
    blockId = tl.program_id(0)
    block_start = blockId*BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    byte_offsets = offsets // 2

    mask = offsets < n_elements
    bytes = tl.load(weight_quant_ptr+byte_offsets, mask=mask)
    is_upper = (offsets % 2) == 0
    nf4_indices = tl.where(is_upper, 
                     (bytes >> 4) & 0x0F,  # Upper 4 bits (when even)
                     bytes & 0x0F)         # Lower 4 bits (when odd)
    K = tl.load(codebook_outer_ptr+nf4_indices, mask=mask)
    outer_block_idx = offsets // block_size_outer
    T = tl.load(absmax_outer_ptr+outer_block_idx, mask=mask)
    F = tl.load(codebook_inner_ptr+T, mask=mask)
    inner_block_idx = outer_block_idx // block_size_inner
    inner_scale = tl.load(absmax_inner_ptr+inner_block_idx, mask=mask)
    values = K * (F * inner_scale)
    tl.store(output_ptr+offsets, values, mask=mask)


def _your_dequantize_nf4(weight: torch.Tensor, quant_state, out_features, in_features):
    n_elements = out_features * in_features
    output = torch.empty(n_elements, dtype=torch.float32, device=weight.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE), )

    _your_dequantize_nf4_kernel[grid](
        output,
        weight.squeeze(1),
        quant_state.absmax,
        quant_state.state2.absmax,
        quant_state.code,
        quant_state.state2.code,
        n_elements,
        quant_state.blocksize,
        quant_state.state2.blocksize,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(out_features, in_features)

def your_dequantize_nf4(weight):
    out_features = weight.out_features
    in_features = weight.in_features
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state, out_features, 
        in_features)