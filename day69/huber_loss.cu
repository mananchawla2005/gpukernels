#include <cuda_runtime.h>
#define e 1e-8

__global__ void huber_kernel(
    const float4* __restrict__ predictions, 
    const float4* __restrict__ targets, 
    float4* __restrict__ output, 
    size_t n, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int i0 = 2 * idx;     
    int i1 = i0 + 1;       

    if (i1 >= n / 4) return; 

    float4 pred0 = __ldg(&predictions[i0]);
    float4 pred1 = __ldg(&predictions[i1]);
    float4 targ0 = __ldg(&targets[i0]);
    float4 targ1 = __ldg(&targets[i1]);

    float4 out0, out1;

    float sub1 = fabsf(pred0.x - targ0.x);
    float sub2 = fabsf(pred0.y - targ0.y);
    float sub3 = fabsf(pred0.z - targ0.z);
    float sub4 = fabsf(pred0.w - targ0.w);
    int cond1 = sub1 < 1.0f;
    int cond2 = sub2 < 1.0f;
    int cond3 = sub3 < 1.0f;
    int cond4 = sub4 < 1.0f;
    out0.x = cond1 * 0.5f * sub1 * sub1 + (!cond1) * (sub1 - 0.5f);
    out0.y = cond2 * 0.5f * sub2 * sub2 + (!cond2) * (sub2 - 0.5f);
    out0.z = cond3 * 0.5f * sub3 * sub3 + (!cond3) * (sub3 - 0.5f);
    out0.w = cond4 * 0.5f * sub4 * sub4 + (!cond4) * (sub4 - 0.5f);

    sub1 = fabsf(pred1.x - targ1.x);
    sub2 = fabsf(pred1.y - targ1.y);
    sub3 = fabsf(pred1.z - targ1.z);
    sub4 = fabsf(pred1.w - targ1.w);
    cond1 = sub1 < 1.0f;
    cond2 = sub2 < 1.0f;
    cond3 = sub3 < 1.0f;
    cond4 = sub4 < 1.0f;
    out1.x = cond1 * 0.5f * sub1 * sub1 + (!cond1) * (sub1 - 0.5f);
    out1.y = cond2 * 0.5f * sub2 * sub2 + (!cond2) * (sub2 - 0.5f);
    out1.z = cond3 * 0.5f * sub3 * sub3 + (!cond3) * (sub3 - 0.5f);
    out1.w = cond4 * 0.5f * sub4 * sub4 + (!cond4) * (sub4 - 0.5f);

    output[i0] = out0;
    output[i1] = out1;
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n)
{
    int total_float4 = n / 4;
    int total_pairs = total_float4 / 2;  

    int blockSize = 1024;
    int gridSize = (total_pairs + blockSize - 1) / blockSize;

    huber_kernel<<<gridSize, blockSize>>>(
        reinterpret_cast<const float4*>(predictions),
        reinterpret_cast<const float4*>(targets),
        reinterpret_cast<float4*>(output),
        n,
        total_pairs);
}
