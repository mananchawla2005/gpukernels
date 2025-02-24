#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#define TILE_WIDTH 16

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < rows) {
        float local_max = -INFINITY;
        for (int i = tid; i < cols; i += blockDim.x) {
            if(input[row * cols + i]>local_max){
                local_max = input[row*cols+i];
            }
        }
        __shared__ float temp_max[256];  // Assuming max block size of 256
        temp_max[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                temp_max[tid] = max(temp_max[tid], temp_max[tid + stride]);
            }
            // sync before next iteration
            __syncthreads();
        }
        
        float global_max = temp_max[0];
        __syncthreads();

        float local_exp_sum = 0.0f;
        for (int i = tid; i < cols; i += blockDim.x) {
            float exp_val = expf(input[row * cols + i] - global_max);
            local_exp_sum += exp_val;
            output[row * cols + i] = exp_val;  // Store intermediate results
        }
        
        __shared__ float total_sum;
        if (tid == 0) total_sum = 0.0f;
        __syncthreads();
        
        atomicAdd(&total_sum, local_exp_sum);
        __syncthreads();
       
        for (int i = tid; i < cols; i += blockDim.x) {
            output[row * cols + i] /= total_sum;
        }
    }
}

__global__ void tiled_matmul_kernel(float* M, float* N, float* P,
                                    int M_rows, int M_cols, int N_cols, int d, bool normalise) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pval = 0.0f;
    int numTiles = (M_cols + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int phase = 0; phase < numTiles; phase++) {
        int m_col = phase * TILE_WIDTH + tx;
        int n_row = phase * TILE_WIDTH + ty;

        if (row < M_rows && m_col < M_cols)
            Mds[ty][tx] = M[row * M_cols + m_col];
        else
            Mds[ty][tx] = 0.0f;
        
        // When normalise==true, we want to load N in transposed form (for Kᵀ)
        if (normalise) {
            if (col < N_cols && n_row < M_cols)
                Nds[ty][tx] = N[col * M_cols + n_row];
            else
                Nds[ty][tx] = 0.0f;
        } else {
            if (n_row < M_cols && col < N_cols)
                Nds[ty][tx] = N[n_row * N_cols + col];
            else
                Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Pval += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (row < M_rows && col < N_cols) {
        if (normalise)
            P[row * N_cols + col] = Pval / sqrtf(d);
        else
            P[row * N_cols + col] = Pval;
    }
}


extern "C" void self_attn(float* q_h, float* attn_out_h, int rows, int cols, int d) {
    float *q_d, *k_d, *v_d, *s_d, *soft_out_d, *attn_out_d;
    int size = rows * cols * sizeof(float);

    cudaMalloc((void**)&q_d, size);
    cudaMalloc((void**)&k_d, size);
    cudaMalloc((void**)&v_d, size);
    cudaMalloc((void**)&s_d, rows*rows*sizeof(float));
    cudaMalloc((void**)&soft_out_d, rows*rows*sizeof(float));
    cudaMalloc((void**)&attn_out_d, size);
    cudaMemcpy(q_d, q_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, q_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, q_h, size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(cols/16.0), ceil(rows/16.0));
    tiled_matmul_kernel<<<gridSize, blockSize>>>(q_d, k_d, s_d, rows, cols, rows, d, true);
    
    int blockSize2 = 256;
    int gridSize2 = rows;
    
    softmax_kernel<<<gridSize2, blockSize2>>>(s_d, soft_out_d, rows, rows);
    dim3 blockSize3(16, 16);
    dim3 gridSize3(ceil(cols/16.0), ceil(rows/16.0));

    tiled_matmul_kernel<<<gridSize3, blockSize3>>>(soft_out_d, v_d, attn_out_d, rows, rows, cols, d, false);
    cudaMemcpy(attn_out_h, attn_out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
    cudaFree(s_d);
    cudaFree(soft_out_d);
    cudaFree(attn_out_d);

}