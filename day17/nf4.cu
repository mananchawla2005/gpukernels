#include <math.h>
#include <stdio.h>

extern "C" {
    typedef enum {
        TYPE_FLOAT16,
    } DataType;

    struct quant_state {
        DataType type;
        float* absmax;
        float* code;
        float* offset = {0};
        int blocksize;
    };
        
    struct nf4 {
        uint8_t* weight;
        quant_state quant_state;
    };
}

float nf4_code[16] = {-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
};
__constant__ float abs_quant_code[256];

__device__ unsigned char decision_nf4(float x)
{

  if(x > 0.03979014977812767f)
    if(x > 0.3893125355243683f) // 1
      if(x > 0.6427869200706482f) // 11
        if(x > 0.8614784181118011f) // 111
          return 0b1111;
        else
          return 0b1110;
      else
        if(x > 0.5016634166240692f) // 110
          return 0b1101;
        else
          return 0b1100;
    else
      if(x > 0.2035212516784668f) // 10
        if(x > 0.2920137718319893f) // 101
          return 0b1011;
        else
          return 0b1010;
      else
        if(x > 0.1202552504837513f) // 100
          return 0b1001;
        else
          return 0b1000;
  else
    if(x > -0.33967943489551544f) // 0
      if(x > -0.13791173323988914f) // 01
        if(x > -0.045525018125772476f) // 011
          return 0b0111;
        else
          return 0b0110;
      else
        if(x > -0.23460740596055984f) // 010
          return 0b0101;
        else
          return 0b0100;
    else
      if(x > -0.6106329262256622f) // 00
        if(x > -0.4599952697753906f) // 001
          return 0b0011;
        else
          return 0b0010;
      else
        if(x > -0.8480964004993439f) // 000
          return 0b0001;
        else
          return 0b0000;
}

__device__ uint8_t pack_nf4(uint8_t nf4_1, uint8_t nf4_2) {
    return (nf4_1 << 4) | (nf4_2 & 0x0F);
}

__global__ void quantize_nf4_block_kernel(const float *input, uint8_t* output, float* absmax, int n){
    int blockSize = blockDim.x;
    int idx = blockIdx.x*blockSize+threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_max[];

    if(idx<n){
        shared_max[tid] = fabsf(input[idx]);
        __syncthreads();
        for (int stride = blockSize / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
            }
            __syncthreads();
        }

        if(tid==0) {
            absmax[blockIdx.x] = shared_max[0];
        }
        __syncthreads();

        if(tid%2==0 && idx<n) {
            uint8_t quantised1 = decision_nf4(input[idx] / shared_max[0]);
            uint8_t quantised2 = 0;
            
            if (idx + 1 < n) {
                quantised2 = decision_nf4(input[idx + 1] / shared_max[0]);
            }
            
            int output_idx = (idx / 2);
            output[output_idx] = pack_nf4(quantised1, quantised2);
        }


        
    }

}

extern "C" nf4 quantize_nf4(const float *input_h, int n, int block_size_outer, int block_size_inner) {
  int blockSize = block_size_outer;
  int gridSize = ceil(n/float(blockSize));
  float *input_d;
  uint8_t* output_d, *output_h;
  float* absmax_d, *absmax_h;

  cudaMallocHost((void**)&absmax_h, gridSize*sizeof(float));
  cudaMallocHost((void**)&output_h, (n/2)*sizeof(uint8_t));
  cudaMalloc((void**)&absmax_d, gridSize*sizeof(float));
  cudaMalloc((void**)&input_d, n*sizeof(float));
  cudaMalloc((void**)&output_d, (n/2)*sizeof(uint8_t));
  cudaMemcpy(input_d, input_h, n*sizeof(float), cudaMemcpyHostToDevice);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  quantize_nf4_block_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(input_d, output_d, absmax_d, n);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
      return {};
  }
  
  cudaMemcpy(output_h, output_d, (n/2)*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(absmax_h, absmax_d, gridSize*sizeof(float), cudaMemcpyDeviceToHost);
  nf4 quant_state;
  quant_state.weight = output_h;
  quant_state.quant_state.type = TYPE_FLOAT16;
  quant_state.quant_state.absmax = absmax_h;
  quant_state.quant_state.blocksize = blockSize;
  quant_state.quant_state.code = nf4_code;
  
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(absmax_d);
  return quant_state;   
}