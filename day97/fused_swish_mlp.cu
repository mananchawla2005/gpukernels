#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

#define BLOCK_SIZE 256
#define TILE_SIZE 128

// SwiGLU activation: Swish(gate) * value
__device__ __forceinline__ float swiglu_activation(float gate, float value) {
    float swish = gate / (1.0f + expf(-gate));
    return swish * value;
}

__global__ void fused_swiglu_mlp_kernel(
    const half* __restrict__ input,       // [batch_seq, hidden_dim]
    const half* __restrict__ gate_weight, // [hidden_dim, intermediate_dim]
    const half* __restrict__ up_weight,   // [hidden_dim, intermediate_dim]
    const half* __restrict__ down_weight, // [intermediate_dim, hidden_dim]
    float* __restrict__ output_f,         // [batch_seq, hidden_dim]
    int batch_seq,
    int hidden_dim,
    int intermediate_dim
) {
    extern __shared__ char smem[];
    half*  input_tile = reinterpret_cast<half*>(smem);
    float* activated  = reinterpret_cast<float*>(smem + hidden_dim * sizeof(half));

    int seq_idx  = blockIdx.x;
    int tile_idx = blockIdx.y;
    int tid      = threadIdx.x;
    if (seq_idx >= batch_seq) return;

    int start_i = tile_idx * TILE_SIZE;
    int end_i   = min(start_i + TILE_SIZE, intermediate_dim);
    int tsize   = end_i - start_i;

    // load input into shared
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE)
        input_tile[i] = input[seq_idx * hidden_dim + i];
    __syncthreads();

    // SwiGLU (read gate/up from global)
    for (int i = tid; i < tsize; i += BLOCK_SIZE) {
        float gsum = 0.f, usum = 0.f;
        int col = start_i + i;
        for (int h = 0; h < hidden_dim; ++h) {
            float iv = __half2float(input_tile[h]);
            int widx = h * intermediate_dim + col;
            gsum += iv * __half2float(gate_weight[widx]);
            usum += iv * __half2float(up_weight  [widx]);
        }
        activated[i] = swiglu_activation(gsum, usum);
    }
    __syncthreads();

    // down‐projection
    for (int h = tid; h < hidden_dim; h += BLOCK_SIZE) {
        float sum = 0.f;
        for (int i = 0; i < tsize; ++i) {
            int gi = start_i + i;
            sum += activated[i] * __half2float(down_weight[gi * hidden_dim + h]);
        }
        atomicAdd(&output_f[seq_idx * hidden_dim + h], sum);
    }
}

int main() {
    int B = 8, S = 512;
    int hidden_dim = 768, intermediate_dim = 3072;
    int batch_seq = B * S;

    size_t in_sz  = batch_seq * hidden_dim;
    size_t gate_sz= hidden_dim * intermediate_dim;
    size_t down_sz= intermediate_dim * hidden_dim;  
    size_t out_sz = in_sz;

    // host buffers
    half *in_h  = new half[in_sz];
    half *gw_h  = new half[gate_sz];
    half *uw_h  = new half[gate_sz];
    half *dw_h  = new half[down_sz];
    half *out_h = new half[out_sz];

    // init with smaller values to prevent overflow
    for (size_t i=0; i<in_sz; ++i) in_h[i] = __float2half((rand()/float(RAND_MAX) - 0.5f) * 0.1f);
    for (size_t i=0; i<gate_sz; ++i) { 
        gw_h[i]=__float2half((rand()/float(RAND_MAX) - 0.5f) * 0.1f); 
        uw_h[i]=__float2half((rand()/float(RAND_MAX) - 0.5f) * 0.1f); 
    }
    for (size_t i=0; i<down_sz; ++i) dw_h[i]=__float2half((rand()/float(RAND_MAX) - 0.5f) * 0.1f);

    // device buffers
    half *in_d, *gw_d, *uw_d, *dw_d, *out_d;
    float* of_d;
    cudaMalloc(&in_d,  in_sz * sizeof(half));  
    cudaMalloc(&gw_d,  gate_sz * sizeof(half)); 
    cudaMalloc(&uw_d,  gate_sz * sizeof(half)); 
    cudaMalloc(&dw_d,  down_sz * sizeof(half)); 
    cudaMalloc(&out_d, out_sz * sizeof(half));  
    cudaMalloc(&of_d,  out_sz * sizeof(float)); 

    cudaMemcpy(in_d, in_h, in_sz * sizeof(half), cudaMemcpyHostToDevice);      // removed checkCuda
    cudaMemcpy(gw_d, gw_h, gate_sz * sizeof(half), cudaMemcpyHostToDevice);    // removed checkCuda
    cudaMemcpy(uw_d, uw_h, gate_sz * sizeof(half), cudaMemcpyHostToDevice);    // removed checkCuda
    cudaMemcpy(dw_d, dw_h, down_sz * sizeof(half), cudaMemcpyHostToDevice);    // removed checkCuda
    cudaMemset(of_d, 0, out_sz * sizeof(float));                               // removed checkCuda

    int tiles = (intermediate_dim + TILE_SIZE -1)/TILE_SIZE;
    dim3 grid(batch_seq, tiles), block(BLOCK_SIZE);
    size_t smem = hidden_dim * sizeof(half)
                + TILE_SIZE  * sizeof(float);

    // warmup
    fused_swiglu_mlp_kernel<<<grid, block, smem>>>(in_d,gw_d,uw_d,dw_d,of_d,batch_seq,hidden_dim,intermediate_dim);
    cudaGetLastError(); cudaDeviceSynchronize(); // removed checkCuda

    // bench
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<100;++i) fused_swiglu_mlp_kernel<<<grid,block,smem>>>(in_d,gw_d,uw_d,dw_d,of_d,batch_seq,hidden_dim,intermediate_dim);
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    float avg_ms = std::chrono::duration<float,std::milli>(t1-t0).count()/100.;

    // convert and copy back on host
    float* of_h = new float[out_sz];
    cudaMemcpy(of_h, of_d, out_sz*sizeof(float), cudaMemcpyDeviceToHost); // removed checkCuda
    for(size_t i=0;i<out_sz;++i) out_h[i]=__float2half(of_h[i]);

    std::cout<<"Avg time: "<<avg_ms<<" ms\n";
    std::cout<<"O[0..2]="<<__half2float(out_h[0])<<","<<__half2float(out_h[1])<<","<<__half2float(out_h[2])<<"\n";

    // cleanup
    delete[] in_h; delete[] gw_h; delete[] uw_h; delete[] dw_h; delete[] out_h; delete[] of_h;
    cudaFree(in_d); cudaFree(gw_d); cudaFree(uw_d); cudaFree(dw_d); cudaFree(out_d); cudaFree(of_d);
    return 0;
}