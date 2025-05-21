#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":"         \
                      << __LINE__ << " -> " << cudaGetErrorString(err) \
                      << std::endl;                                     \
            std::exit(EXIT_FAILURE);                                    \
        }                                                                \
    } while (0)

// append_kv_cache kernel
__global__
void append_kv_cache(
    const float* __restrict__ new_K,
    const float* __restrict__ new_V,
    float*       __restrict__ k_cache,
    float*       __restrict__ v_cache,
    int B, int H, int D, int L_max, int t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * D;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    int b = tmp / H;

    int cache_base = ((b*H + h)*L_max + t)*D + d;
    k_cache[cache_base] = new_K[idx];
    v_cache[cache_base] = new_V[idx];
}

// compute_attention_scores kernel
__global__
void compute_attention_scores(
    const float* __restrict__ Q,
    const float* __restrict__ k_cache,
    float*       __restrict__ scores,
    int B, int H, int D, int t, int L_max)
{
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * t;
    if (linear_idx >= total) return;

    int pos = linear_idx % t;
    int tmp = linear_idx / t;
    int h = tmp % H;
    int b = tmp / H;

    float acc = 0.0f;
    int q_base = ((b*H + h)*D);
    int k_base = ((b*H + h)*L_max + pos)*D;
    for (int i = 0; i < D; ++i) {
        acc += Q[q_base + i] * k_cache[k_base + i];
    }
    // Scale by 1/sqrt(D)
    scores[linear_idx] = acc / sqrtf(D);
}

int main() {
    const int B = 1;       // batch size
    const int H = 1;       // num heads
    const int D = 4;       // head dimension
    const int L_max = 8;   // max sequence length
    const int steps = 4;   // number of append steps

    std::vector<float> h_new_K(B*H*D), h_new_V(B*H*D), h_Q(B*H*D);

    for (int i = 0; i < B*H*D; ++i) {
        h_new_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_new_V[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Q[i]     = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_new_K, *d_new_V, *d_Q;
    float *d_k_cache, *d_v_cache, *d_scores;

    CUDA_CHECK(cudaMalloc(&d_new_K, sizeof(float)*B*H*D));
    CUDA_CHECK(cudaMalloc(&d_new_V, sizeof(float)*B*H*D));
    CUDA_CHECK(cudaMalloc(&d_Q,     sizeof(float)*B*H*D));
    CUDA_CHECK(cudaMalloc(&d_k_cache,sizeof(float)*B*H*L_max*D));
    CUDA_CHECK(cudaMalloc(&d_v_cache,sizeof(float)*B*H*L_max*D));
    CUDA_CHECK(cudaMalloc(&d_scores,sizeof(float)*B*H*steps));

    CUDA_CHECK(cudaMemcpy(d_new_K, h_new_K.data(), sizeof(float)*B*H*D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new_V, h_new_V.data(), sizeof(float)*B*H*D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q,     h_Q.data(),     sizeof(float)*B*H*D, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_k_cache, 0, sizeof(float)*B*H*L_max*D));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, sizeof(float)*B*H*L_max*D));

    int t = 0;
    int threads = 256;
    int blocks_append = (B*H*D + threads - 1) / threads;

    // Loop over steps: append and compute attention
    for (; t < steps; ++t) {
    for (int i = 0; i < B*H*D; ++i) {
        h_new_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_new_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_new_K, h_new_K.data(), sizeof(float)*B*H*D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new_V, h_new_V.data(), sizeof(float)*B*H*D, cudaMemcpyHostToDevice));

    append_kv_cache<<<blocks_append, threads>>>(
        d_new_K, d_new_V,
        d_k_cache, d_v_cache,
        B, H, D, L_max, t);
        CUDA_CHECK(cudaDeviceSynchronize());

        int total_scores = B*H*(t+1);
        int blocks_score = (total_scores + threads - 1) / threads;
        compute_attention_scores<<<blocks_score, threads>>>(
            d_Q, d_k_cache, d_scores,
            B, H, D, t+1, L_max);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy and print scores for this step
        std::vector<float> h_scores(total_scores);
        CUDA_CHECK(cudaMemcpy(h_scores.data(), d_scores,
                              sizeof(float)*total_scores,
                              cudaMemcpyDeviceToHost));
        std::cout << "Step " << t << " scores: ";
        for (float v : h_scores) std::cout << v << " ";
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_new_K);
    cudaFree(d_new_V);
    cudaFree(d_Q);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_scores);

    return 0;
}
