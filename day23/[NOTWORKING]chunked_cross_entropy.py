import torch
import triton
import triton.language as tl

def chunked_cross_entropy(hidden_states: torch.Tensor, # [batch_size, seq_len, hidden_size]
                         targets: torch.Tensor, # [batch_size, seq_len]
                         hidden_size: int, 
                         vocab_size: int, 
                         chunk_size: int
                         ):
    embeddings =  torch.randn(vocab_size, hidden_size)
    bias = torch.zeros(vocab_size)
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    flat_targets = targets.reshape(-1)
    num_rows = flat_hidden.size(0)
    bs = triton.cdiv(vocab_size, chunk_size)
    loss = torch.zeros(flat_targets.size(0))
    chunked_cross_entropy_kernel[bs](flat_hidden, flat_targets, loss, embeddings, bias, hidden_size, vocab_size, chunk_size, bs, num_rows)
    return -loss.mean()

@triton.jit
def chunked_cross_entropy_kernel(flat_hidden, flat_targets, loss, embeddings, bias, hidden_size: tl.constexpr, vocab_size: tl.constexpr, chunk_size: tl.constexpr, bs: tl.constexpr, num_rows: tl.constexpr): 
    blockId = tl.program_id(0)
    chunk_start = blockId*chunk_size
    chunk_end = tl.minimum(chunk_start+chunk_size, vocab_size)
    offsets = tl.arange(0, chunk_size)
    mask = offsets < (chunk_end - chunk_start)
    chunk_offsets = chunk_start + tl.where(mask, offsets, 0)
    chunk_embeddings = tl.load(embeddings + chunk_offsets[:, None]*hidden_size+tl.arange(0, hidden_size)[None, :], mask=mask[:, None])
    chunk_bias = tl.load(bias+chunk_offsets, mask=mask)
    
    row_indices = tl.arange(0, num_rows)
    hidden_block = tl.load(flat_hidden + row_indices[:, None]*hidden_size + tl.arange(0, hidden_size)[None, :])
    chunk_logits = tl.dot(hidden_block, tl.permute(chunk_embeddings, (1, 0))) + chunk_bias
    target_values = tl.load(flat_targets + row_indices)
    chunk_mask = (target_values >= chunk_start) & (target_values < chunk_end)
    chunk_relative_indices = (tl.where(chunk_mask, target_values, 0) - chunk_start).to(tl.int32)
    target_logits = tl.load(chunk_logits + row_indices[:, None] * (chunk_end - chunk_start)
                        + chunk_relative_indices[:, None],
                        mask=chunk_mask[:, None])
    max_logits = tl.max(chunk_logits, 1)
    log_sum_exp = max_logits + tl.log(tl.sum(tl.exp(chunk_logits - max_logits[:, None]), 1))
    tl.store(loss, target_logits-log_sum_exp, mask=chunk_mask)

batch_size = 2
sequence_length = 128
hidden_size = 128
vocab_size = 51200
chunk_size = 1024

# Create sample data
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
targets = torch.randint(0, vocab_size, (batch_size, sequence_length))

# Initialize the model
loss = chunked_cross_entropy(hidden_states, targets, hidden_size, vocab_size, chunk_size)

print(f"Loss: {loss.item()}")