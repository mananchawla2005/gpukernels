#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// --------------------------------------------------
// Normal FP16 ReLU (1 element / thread)
__global__ void relu_f16_kernel(const half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = __hmax(__float2half(0.0f), x[idx]);
  }
}

// Vectorized FP16 ReLU (8 elements / thread)
__global__ void relu_f16x8_kernel(const half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // load up to 4 half2's (8 halves):
  half2 r0 = reinterpret_cast<const half2*>(x + idx)[0];
  half2 r1 = reinterpret_cast<const half2*>(x + idx)[1];
  half2 r2 = reinterpret_cast<const half2*>(x + idx)[2];
  half2 r3 = reinterpret_cast<const half2*>(x + idx)[3];
  // apply ReLU on both lanes
  const half2 zero2 = __float2half2_rn(0.0f);
  r0 = __hmax2(zero2, r0);
  r1 = __hmax2(zero2, r1);
  r2 = __hmax2(zero2, r2);
  r3 = __hmax2(zero2, r3);
  int remaining = N - idx;
  if (remaining >= 8) {
    auto out = reinterpret_cast<half2*>(y + idx);
    out[0] = r0; out[1] = r1; out[2] = r2; out[3] = r3;
  } else {
    // handle tail
    half tmp[8];
    reinterpret_cast<half2*>(tmp)[0] = r0;
    reinterpret_cast<half2*>(tmp)[1] = r1;
    reinterpret_cast<half2*>(tmp)[2] = r2;
    reinterpret_cast<half2*>(tmp)[3] = r3;
    for (int i = 0; i < remaining; ++i)
      y[idx + i] = tmp[i];
  }
}

// host launcher for simplicity:
void launch_relu(const half* d_x, half* d_y, int N,
                 void (*kernel)(const half*, half*, int),
                 int vec_factor)
{
  int threads = 256;
  int elems_per_thread = vec_factor;
  int blocks = (N + threads * elems_per_thread - 1) / (threads * elems_per_thread);
  kernel<<<blocks, threads>>>(d_x, d_y, N);
}

int main() {
  const int N = 20;
  // host arrays
  float h_in[N];
  for (int i = 0; i < N; ++i)
    h_in[i] = (i % 5 == 0) ? -float(i) : float(i);  // mix pos/neg

  // allocate device
  half *d_x, *d_y1, *d_y8;
  cudaMalloc(&d_x, N * sizeof(half));
  cudaMalloc(&d_y1, N * sizeof(half));
  cudaMalloc(&d_y8, N * sizeof(half));

  // convert & copy input to half
  half tmp[N];
  for (int i = 0; i < N; ++i) tmp[i] = __float2half(h_in[i]);
  cudaMemcpy(d_x, tmp, N * sizeof(half), cudaMemcpyHostToDevice);

  // run normal ReLU
  launch_relu(d_x, d_y1, N, relu_f16_kernel, 1);
  // run vector ReLU
  launch_relu(d_x, d_y8, N, relu_f16x8_kernel, 8);

  // copy back
  half out1[N], out8[N];
  cudaMemcpy(out1, d_y1, N * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(out8, d_y8, N * sizeof(half), cudaMemcpyDeviceToHost);

  // print
  printf("i |   in   | ReLU1 | ReLU8\n");
  printf("---+--------+-------+-------\n");
  for (int i = 0; i < N; ++i) {
    printf("%2d | %6.2f | %6.2f | %6.2f\n",
           i, h_in[i],
           __half2float(out1[i]),
           __half2float(out8[i]));
  }

  // cleanup
  cudaFree(d_x);
  cudaFree(d_y1);
  cudaFree(d_y8);
  return 0;
}
