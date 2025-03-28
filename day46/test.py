import torch
import math
import torch.nn as nn
import gpu_kernels
from typing import Optional

class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position up to
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as embed_dim // num_heads.
        max_seq_len (int): Maximum expected sequence length for the model.
        base (int): The base of the geometric progression used to compute the rotation angles.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor containing the token position ids.
                                          Default: None.
        Returns:
            Tensor: output tensor with RoPE applied.
        """
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, x_shaped.size(1), 1, x_shaped.size(3), 2)
        x_rotated = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0] + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_rotated.flatten(3)
        return x_out.type_as(x)

if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch = 2
    seq_len = 16
    head_dim = 64
    n_heads = 1
    
    x_ref = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float32)
    x_kernel = x_ref.clone()
    
    out_kernel = torch.empty_like(x_kernel)
    
    gpu_kernels.rope(x_kernel, out_kernel)
    
    rope_module = RotaryPositionalEmbeddings(dim=head_dim)
    out_ref = rope_module(x_ref)
    print("Kernel:", out_kernel)
    print("Referencee:", out_ref)
    max_diff = (out_kernel - out_ref).abs().max().item()
    print("Max difference between custom kernel and PyTorch:", max_diff)
    
    assert max_diff < 1e-5, "Results differ too much!"