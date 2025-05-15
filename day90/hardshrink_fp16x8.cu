#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define LAMBD 0.5f

__device__ __forceinline__ half hardshrink_half(half x) {
    return (x > __float2half(LAMBD) || x < __float2half(-LAMBD))
         ? x
         : __float2half(0.f);
}

__global__ void hardshrink_f16_kernel(const half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = hardshrink_half(x[idx]);
    }
}

__global__ void hardshrink_f16x8_kernel(const half *x, half *y, int N) {
    int base = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    // load 4 × half2 = 8 halves
    half2 r0 = reinterpret_cast<const half2*>(x + base)[0];
    half2 r1 = reinterpret_cast<const half2*>(x + base)[1];
    half2 r2 = reinterpret_cast<const half2*>(x + base)[2];
    half2 r3 = reinterpret_cast<const half2*>(x + base)[3];

    // apply hardshrink on both lanes
    r0.x = hardshrink_half(r0.x);  r0.y = hardshrink_half(r0.y);
    r1.x = hardshrink_half(r1.x);  r1.y = hardshrink_half(r1.y);
    r2.x = hardshrink_half(r2.x);  r2.y = hardshrink_half(r2.y);
    r3.x = hardshrink_half(r3.x);  r3.y = hardshrink_half(r3.y);

    int rem = N - base;
    if (rem >= 8) {
        auto out = reinterpret_cast<half2*>(y + base);
        out[0] = r0;  out[1] = r1;
        out[2] = r2;  out[3] = r3;
    } else {
        // write tail element-wise
        half tmp[8];
        reinterpret_cast<half2*>(tmp)[0] = r0;
        reinterpret_cast<half2*>(tmp)[1] = r1;
        reinterpret_cast<half2*>(tmp)[2] = r2;
        reinterpret_cast<half2*>(tmp)[3] = r3;
        for (int i = 0; i < rem; ++i)
            y[base + i] = tmp[i];
    }
}

void launch_kernel(const half* d_x, half* d_y, int N,
                   void(*ker)(const half*, half*, int), int elems_per_thread)
{
    int threads = 256;
    int blocks = (N + threads * elems_per_thread - 1) / (threads * elems_per_thread);
    ker<<<blocks, threads>>>(d_x, d_y, N);
}

int main() {
    const int N = 20;
    float h_in[N];
    for (int i = 0; i < N; ++i)
        h_in[i] = (i % 5 == 0) ? -float(i) : float(i);

    half *d_x, *d_y1, *d_y8;
    cudaMalloc(&d_x,  N * sizeof(half));
    cudaMalloc(&d_y1, N * sizeof(half));
    cudaMalloc(&d_y8, N * sizeof(half));

    half tmp[N];
    for (int i = 0; i < N; ++i) tmp[i] = __float2half(h_in[i]);
    cudaMemcpy(d_x, tmp, N * sizeof(half), cudaMemcpyHostToDevice);

    // run both
    launch_kernel(d_x, d_y1, N, hardshrink_f16_kernel, 1);
    launch_kernel(d_x, d_y8, N, hardshrink_f16x8_kernel, 8);

    // retrieve results
    half out1[N], out8[N];
    cudaMemcpy(out1, d_y1, N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(out8, d_y8, N * sizeof(half), cudaMemcpyDeviceToHost);

    // print comparison
    printf(" i |   in   | hard1  | hard8\n");
    printf("----+--------+--------+--------\n");
    for (int i = 0; i < N; ++i) {
        printf("%2d | %6.2f | %6.2f | %6.2f\n",
               i, h_in[i],
               __half2float(out1[i]),
               __half2float(out8[i]));
    }

    cudaFree(d_x);
    cudaFree(d_y1);
    cudaFree(d_y8);
    return 0;
}
