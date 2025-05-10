#include <cuda_runtime.h>

__global__ void lower_triangular_matmul_kernel(const float* mat_a, const float* mat_b, float* mat_c, int dim_n) {
    int idx_row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_row >= dim_n || idx_col >= dim_n) return;

    if (idx_col <= idx_row) {
        float val_sum = 0.0f;
        // Only k between col and row matters
        for (int idx_k = idx_col; idx_k <= idx_row; ++idx_k) {
            val_sum += mat_a[idx_row * dim_n + idx_k] * mat_b[idx_k * dim_n + idx_col];
        }
        mat_c[idx_row * dim_n + idx_col] = val_sum;
    } else {
        mat_c[idx_row * dim_n + idx_col] = 0.0f;
    }
}

// mat_a, mat_b, mat_c are device pointers
extern "C" void lower_triangular_matmul_launcher(const float* mat_a, const float* mat_b, float* mat_c, size_t dim_n) {    
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((dim_n + threads_per_block.x - 1) / threads_per_block.x,
                    (dim_n + threads_per_block.y - 1) / threads_per_block.y);

    lower_triangular_matmul_kernel<<<num_blocks, threads_per_block>>>(mat_a, mat_b, mat_c, dim_n);
}