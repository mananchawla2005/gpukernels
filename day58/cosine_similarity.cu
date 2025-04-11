#include <cuda_runtime.h>
#define e 1e-8

__global__ void cosine_kernel(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int total = n;
    if(row>=total){
        return;
    }
    float pred_sum = 0.0f;
    float target_sum = 0.0f;
    float dot_prod = 0.0f;
    for(int c=0;c<d;c++){
        float pred = predictions[row*d+c];
        pred_sum+=pred*pred;
        float target = targets[row*d+c];
        target_sum+=target*target;
        dot_prod += targets[row*d+c] * predictions[row*d+c];
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