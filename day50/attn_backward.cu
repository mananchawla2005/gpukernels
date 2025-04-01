#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#define TILE_WIDTH 16

__global__ void softmax_backward_kernel(float* grad_output, float* softmax_output, 
                                      float* grad_input, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < rows) {
        // First compute sum of (grad_output * softmax_output)
        float sum = 0.0f;
        for (int i = tid; i < cols; i += blockDim.x) {
            sum += grad_output[row * cols + i] * softmax_output[row * cols + i];
        }
        
        __shared__ float total_sum;
        if (tid == 0) total_sum = 0.0f;
        __syncthreads();
        
        atomicAdd(&total_sum, sum);
        __syncthreads();
        
        for (int i = tid; i < cols; i += blockDim.x) {
            int idx = row * cols + i;
            grad_input[idx] = softmax_output[idx] * 
                (grad_output[idx] - total_sum);
        }
    }
}

__global__ void tiled_matmul_kernel(float* M, float* N, float* P,
                                    int M_rows, int M_cols, int N_cols, bool transpose_m, bool transpose_n, int d=1, bool normalise=false) {
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

        if(transpose_m) {
            if (row < M_cols && m_col < M_rows)
                Mds[ty][tx] = M[m_col * M_cols + row];
            else
                Mds[ty][tx] = 0.0f;
        }
        else {

            if (row < M_rows && m_col < M_cols)
                Mds[ty][tx] = M[row * M_cols + m_col];
            else
                Mds[ty][tx] = 0.0f;
        }
        
        if (transpose_n) {
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


extern "C" void self_attn_backward(float* grad_output_h, float* q_h, float* k_h, float* v_h,
                                    float* attn_weights_h,
                                    float* grad_q_h, float* grad_k_h, float* grad_v_h,
                                    int rows, int cols, int d) {
    float *grad_output_d, *q_d, *k_d, *v_d;
    float *s_d, *p_d, *grad_s_d, *grad_p_d, *grad_q_d, *grad_k_d, *grad_v_d;
    int size = rows * cols * sizeof(float);
    int attn_size = rows * rows * sizeof(float);
    cudaMalloc((void**)&grad_output_d, size);
    cudaMalloc((void**)&q_d, size);
    cudaMalloc((void**)&k_d, size);
    cudaMalloc((void**)&v_d, size);
    cudaMalloc((void**)&s_d, attn_size);
    cudaMalloc((void**)&p_d, attn_size);
    cudaMalloc((void**)&grad_p_d, attn_size);
    cudaMalloc((void**)&grad_s_d, attn_size);
    cudaMalloc((void**)&grad_q_d, size);
    cudaMalloc((void**)&grad_k_d, size);
    cudaMalloc((void**)&grad_v_d, size);

    cudaMemcpy(grad_output_d, grad_output_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(q_d, q_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, attn_weights_h, attn_size, cudaMemcpyHostToDevice);
    // 1. Compute dV = P^T × dO
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_WIDTH - 1) / TILE_WIDTH);
    tiled_matmul_kernel<<<dimGrid, dimBlock>>>(p_d,   //  Matrix M (P^T) 
                                                grad_output_d, // Matrix N (dO) 
                                                grad_v_d,    // Output (dV)
                                                rows,       // M_rows
                                                rows,       // M_cols (=rows for P^T)
                                                cols,       // N_cols
                                                true, false);      
    // 2. Compute dP = dO × V^T
    dimGrid.x = (rows + TILE_WIDTH - 1) / TILE_WIDTH;
    dimGrid.y = (rows + TILE_WIDTH - 1) / TILE_WIDTH;
    tiled_matmul_kernel<<<dimGrid, dimBlock>>>(grad_output_d, // Matrix M (dO)
                                                v_d,           // Matrix N (V)
                                                grad_p_d,      // Output (dP)
                                                rows,          // M_rows
                                                cols,          // M_cols
                                                rows,          // N_cols (rows for V^T)
                                                false, true);  
    // 3. Compute dS using softmax_backward_kernel
    dim3 softmaxBlock(256);  
    dim3 softmaxGrid(rows);  
    softmax_backward_kernel<<<softmaxGrid, softmaxBlock>>>(
        grad_p_d,    // gradient from previous step (dP)
        p_d,         // softmax output from forward pass
        grad_s_d,    // output gradient (dS)
        rows,        // number of rows
        rows         // number of columns (P is rows×rows)
    );
    // 4. Compute dQ and dK using scaled dot-product
    // For dQ: Computing dQ = (dS × K)/√d
    // For dK: Computing dK = (dS^T × Q)/√d
    dimGrid.x = (cols + TILE_WIDTH - 1) / TILE_WIDTH;
    dimGrid.y = (rows + TILE_WIDTH - 1) / TILE_WIDTH;
    tiled_matmul_kernel<<<dimGrid, dimBlock>>>(
        grad_s_d,  // Matrix M (dS)
        k_d,       // Matrix N (K)
        grad_q_d,  // Output (dQ)
        rows,      // M_rows
        rows,      // M_cols
        cols,      // N_cols
        false,       
        false,
        d,   // d value for scaling
        true
    );
    tiled_matmul_kernel<<<dimGrid, dimBlock>>>(
        grad_s_d,  // Matrix M (dS^T)
        q_d,       // Matrix N (Q)
        grad_k_d,  // Output (dK)
        rows,      // M_rows
        rows,      // M_cols
        cols,      // N_cols
        true,
        false,
        d,
        true
    );


    cudaMemcpy(grad_q_h, grad_q_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_k_h, grad_k_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_v_h, grad_v_d, size, cudaMemcpyDeviceToHost);

    cudaFree(grad_output_d);
    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
    cudaFree(s_d);
    cudaFree(p_d);
    cudaFree(grad_s_d);
    cudaFree(grad_q_d);
    cudaFree(grad_k_d);
    cudaFree(grad_v_d);
    cudaFree(grad_p_d);
}