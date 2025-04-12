#include <cuda_runtime.h>
#define e 1e-8

__global__ void cosine_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, size_t n, size_t d) {
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int total = n;
    if(row>=total){
        return;
    }
    float pred_sum = 0.0f;
    float target_sum = 0.0f;
    float dot_prod = 0.0f;
    
    #pragma unroll
    for(int c=0;c<d;c++){
        float pred = __ldg(&predictions[row*d + c]);
        pred_sum = fmaf(pred, pred, pred_sum);
        float target = __ldg(&targets[row*d + c]);
        target_sum = fmaf(target, target, target_sum);
        dot_prod = fmaf(pred, target, dot_prod); 
    }
    float cos_sim = (dot_prod)/(max(e, sqrtf(pred_sum))*max(e, sqrtf(target_sum)));
    output[row] = 1-cos_sim;

}

// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {    
    int total = n;
    int blockSize = 256;
    int gridSize = ceil(total/float(blockSize));
    cosine_kernel<<<gridSize, blockSize>>>(predictions, targets, output, n, d);
}