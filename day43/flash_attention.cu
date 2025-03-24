#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define D 64  // Must match actual d dimension
#define TILE_SIZE 16

__global__ void flash_attention_kernel(float* q_d, float* k_d, float* v_d, float* o_d, 
                                      float* l_d, float* m_d, int n, int d) {
    __shared__ float q_s[TILE_SIZE][D];
    __shared__ float k_s[TILE_SIZE][D];
    __shared__ float v_s[TILE_SIZE][D];
    __shared__ float s_s[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int row = bx * TILE_SIZE + tx;
    
    if(row >= n) return;

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_local[D] = {0};

    for(int k = 0; k < d; k++) {
        q_s[tx][k] = q_d[row * d + k];
    }

    const int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    const float scale = 1.0f / sqrtf(d);

    for(int t = 0; t < num_tiles; ++t) {
        int col = t * TILE_SIZE + tx;
        if(col < n) {
            for(int k = 0; k < d; k++) {
                k_s[tx][k] = k_d[col * d + k];
                v_s[tx][k] = v_d[col * d + k];
            }
        } else {
            for(int k = 0; k < d; k++) {
                k_s[tx][k] = 0.0f;
                v_s[tx][k] = 0.0f;
            }
        }
        __syncthreads();

        float max_val = -INFINITY;
        for(int i = 0; i < TILE_SIZE; ++i) {
            float sum = 0.0f;
            for(int k = 0; k < d; ++k) {
                sum += q_s[tx][k] * k_s[i][k];
            }
            s_s[tx][i] = (t * TILE_SIZE + i < n) ? sum * scale : -INFINITY;
            max_val = fmaxf(max_val, s_s[tx][i]);
        }

        float exp_sum = 0.0f;
        for(int i = 0; i < TILE_SIZE; ++i) {
            s_s[tx][i] = expf(s_s[tx][i] - max_val);
            exp_sum += (t * TILE_SIZE + i < n) ? s_s[tx][i] : 0.0f;
        }

        float m_new = fmaxf(m_prev, max_val);
        float l_new = expf(m_prev - m_new) * l_prev + exp_sum * expf(max_val - m_new);
        
        for(int c = 0; c < d; ++c) {
            float p_sum = 0.0f;
            for(int i = 0; i < TILE_SIZE; ++i) {
                p_sum += s_s[tx][i] * v_s[i][c];
            }
            o_local[c] = (expf(m_prev - m_new) * l_prev * o_local[c] + 
                          expf(max_val - m_new) * p_sum) / l_new;
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    for(int c = 0; c < d; ++c) {
        o_d[row * d + c] = o_local[c];
    }
    m_d[row] = m_prev;
    l_d[row] = l_prev;
}

extern "C" void flash_attention(float* q_h, float* k_h, float* v_h, float* o_h, int n, int d) {
    float *q_d, *k_d, *v_d, *o_d, *l_d, *m_d;
    int size = n*d*sizeof(float);
    cudaMalloc((void**)&q_d, size);
    cudaMalloc((void**)&k_d, size);
    cudaMalloc((void**)&v_d, size);
    cudaMemcpy(q_d, q_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&o_d, size);
    cudaMemset(o_d, 0, size);
    cudaMalloc((void**)&l_d, sizeof(float)*n);
    cudaMemset(l_d, 0, sizeof(float)*n);
    cudaMalloc((void**)&m_d, sizeof(float)*n);
    cudaMemset(m_d, -INFINITY, sizeof(float)*n);
    int n_tile= ceil(n/float(TILE_SIZE)); // no. of tiles
    dim3 blockSize(TILE_SIZE);
    dim3 gridSize(n_tile);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    flash_attention_kernel<<<gridSize, blockSize>>>(q_d, k_d, v_d, o_d, l_d, m_d, n, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f ms\n", elapsedTime);
    
    cudaMemcpy(o_h, o_d, size, cudaMemcpyDeviceToHost);
    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
    cudaFree(l_d);
    cudaFree(m_d);
}