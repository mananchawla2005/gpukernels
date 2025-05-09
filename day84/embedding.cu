__global__ void embedding_forward_kernel(
    const float* __restrict__ embeddings,  // [vocab_size * embedding_dim]
    const int* __restrict__ indices,       // [n_elements]
    float* __restrict__ output,            // [n_elements * embedding_dim]
    int n_elements,
    int embedding_dim
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;  // row
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;    // column

    if (token_idx < n_elements && dim_idx < embedding_dim) {
        int token_id = indices[token_idx];
        float val = embeddings[token_id * embedding_dim + dim_idx];
        output[token_idx * embedding_dim + dim_idx] = val;
    }
}