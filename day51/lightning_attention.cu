#include <cuda_runtime.h>
#define B 16
#define D 64

__global__ void lightning_attention_kernel(float *q, float *k, float *v, float *o, float *m, int n, int d)
{
    __shared__ float q_s[B*D];
    __shared__ float k_s[B*D];
    __shared__ float v_s[B*D];
    __shared__ float kv[D*D];

    int t = blockIdx.x;
    int tid = threadIdx.x;
    int span = t*B*D;

    for (int i = tid; i < B*D; i+=blockDim.x)
    {
        if(span+i<n*D) {
            q_s[i] = q[span+i];
            k_s[i] = k[span+i];
            v_s[i] = v[span+i];
        }
        else {
            q_s[i] = 0.0f;
            k_s[i] = 0.0f;
            v_s[i] = 0.0f;
        }
    }
    __syncthreads();

    for (int i = tid; i < D*D; i+=blockDim.x)
    {
        kv[i] = 0.0f;
    }
    __syncthreads();

    // O_Intra = [(QtKt^T).M]Vt
    __shared__ float o_intra[B*D];
    for (int i = tid; i < B*D; i+=blockDim.x)
    {
        o_intra[i] = 0.0f;
    }

    __syncthreads();
    for (int i = tid; i < B*B; i+=blockDim.x)
    {
        int row = i/B;
        int col = i%B;
        float accumulator = 0.0f;
        for (int j = 0; j < D; j++)
        {
            accumulator += q_s[row*D+j]*k_s[col*D+j];
        }
        accumulator = accumulator*m[row*B+col];
        for (int j = 0; j < D; j++)
        {
            float result = accumulator*v_s[col*D+j];
            atomicAdd(&o_intra[row*D+j], result);
        }        
        
    }
    __syncthreads();

    // O_Inter = Qt(KV)
    for (int i = tid; i < D * D; i += blockDim.x) {
        int row = i / D;
        int col = i % D;
        float sum = 0.0f;
        for (int s = 0; s < B; s++) {
            sum += k_s[s * D + row] * v_s[s * D + col];
        }
        atomicAdd(&kv[i], sum);
    }
    __syncthreads();

    __shared__ float o_inter[B*D];

    for (int i = tid; i < B*D; i+=blockDim.x)
    {
        o_inter[i] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < B*D; i+=blockDim.x)
    {
        int row = i/D;
        int col = i%D;
        float accumulator = 0.0f;
        for (int j = 0; j < D; j++)
        {
            accumulator += q_s[row*D+j]*kv[j*D+col];
        }
        o_inter[row*D+col] = accumulator;
        
    }
    __syncthreads();

    // Ot = o_intra + o_inter
    for (int i = tid; i < B * D; i += blockDim.x) {
        if (span+i < n*d) {
            o[span+i] = (o_intra[i]+o_inter[i])/sqrtf(d);
        }
    }

}

extern "C" void lightning_attention(float *q_h, float *k_h, float *v_h, float *o_h, int n, int d)
{
    float *q_d, *k_d, *v_d, *o_d, *m_d, *m_h;
    int size = n * d * sizeof(float);
    int maskSize = B * B * sizeof(float);
    cudaMallocHost(&m_h, maskSize);
    for (int i = 0; i < B * B; i++)
    {
        m_h[i] = (i % B <= i / B) ? 1.0f : 0.0f; // mask condition
        // The condition ensures that
        // column ≤ row
        // which masks out future tokens.
        // This creates a lower triangular matrix with 1s in the valid positions and 0s where future tokens should be ignored.
    }
    
    cudaMalloc((void**)&q_d, size);
    cudaMalloc((void**)&k_d, size);
    cudaMalloc((void**)&v_d, size);
    cudaMalloc((void**)&o_d, size);
    cudaMalloc((void**)&m_d, maskSize);

    cudaMemcpy(q_d, q_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(o_d, o_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, maskSize, cudaMemcpyHostToDevice);

    int numBlocks = ceil(n / float(B));
    dim3 blockSize(256);
    dim3 gridSize(numBlocks);

    lightning_attention_kernel<<<gridSize, blockSize>>>(q_d, k_d, v_d, o_d, m_d, n, d);
    cudaMemcpy(o_h, o_d, size, cudaMemcpyDeviceToHost);
    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
    cudaFree(o_d);
    cudaFree(m_d);
    cudaFreeHost(m_h);
}