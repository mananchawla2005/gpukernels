#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// #define M 700000 // 786432
#define D 64 // embedding dimension known at compile time
#define TILE_SIZE 16
__global__ void flash_attention_kernel(float* q_d, float* k_d, float* v_d, float* o_d, float* l_d, float* m_d, int n, int d) {
    __shared__ float q_s[TILE_SIZE][D];
    __shared__ float k_s[TILE_SIZE][D];
    __shared__ float v_s[TILE_SIZE][D];
    __shared__ float s_s[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int row = bx*TILE_SIZE+tx;
    float m_i = -INFINITY;
    float l_i = 0; 
    if(row>=n){
        return;
    }
    for (size_t j = 0; j < gridDim.x; j++)
    {
        int col = j*TILE_SIZE+tx;
        for (size_t k = 0; k < d; k++)
        {
            if(col<n) {
                k_s[tx][k] = k_d[col * d + k]; // kj
                v_s[tx][k] = v_d[col * d + k]; // vj
            }
            else {
                k_s[tx][k] = 0.0f;
                v_s[tx][k] = 0.0f;
            }
        }
            
        __syncthreads();

        for (size_t i = 0; i < gridDim.x; i++)
        {
            int q_row = i*TILE_SIZE+tx;
            if(tx<TILE_SIZE) {
                for (size_t k = 0; k < d; k++)
                {
                    if(q_row<n) {
                        q_s[tx][k] = q_d[q_row * d + k]; // qi
                    }
                    else {
                        q_s[tx][k] = 0.0f;
                    }
                }        
            }
            __syncthreads();

            if (tx >= TILE_SIZE || q_row >= n) {
                continue;
            }
            float m_new, l_new, m_curr, l_curr;
            for (size_t c = 0; c < TILE_SIZE; c++)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < d; k++)
                {
                    sum+=q_s[tx][k]*k_s[c][k];
                }
                if(j*TILE_SIZE+c<n){
                    s_s[tx][c] = sum/sqrtf(d);
                }
                else {
                    s_s[tx][c] = -INFINITY;
                }
                
                m_curr = fmaxf(m_curr, sum/sqrtf(d)); // row max calculation
            }
            for (size_t c = 0; c < TILE_SIZE; ++c) {
                s_s[tx * TILE_SIZE + c] = expf(s_s[tx * TILE_SIZE + c] - m_curr);
                l_curr += s_s[tx * TILE_SIZE + c];
            }
            
    
        }
        
        
    }
    
    

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
    flash_attention_kernel<<<gridSize, blockSize>>>(q_d, k_d, v_d, o_d, l_d, m_d, n, d);
    cudaMemcpy(o_h, o_d, size, cudaMemcpyDeviceToHost);
    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
    cudaFree(l_d);
    cudaFree(m_d);
}