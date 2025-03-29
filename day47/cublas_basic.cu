#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#define N (275)
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

//  SGEMM  performs matrix-matrix operation

// C := alpha*op( A )*op( B ) + beta*C,

int main(int argc, char **argv)
{
    cublasStatus_t status;
    float* A_h;
    float* B_h;
    float* C_h;
    float* C_h_ref;

    float* A_d = 0;
    float* B_d = 0;
    float* C_d = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    A_h = (float *)malloc(n2 * sizeof(A_h[0]));
    B_h = (float *)malloc(n2 * sizeof(B_h[0]));
    C_h = (float *)malloc(n2 * sizeof(C_h[0]));

    for (i = 0; i < n2; i++)
    {
        A_h[i] = rand() / (float)RAND_MAX;
        B_h[i] = rand() / (float)RAND_MAX;
        C_h[i] = rand() / (float)RAND_MAX;
    }
    cudaMalloc((void **)&A_d, n2 * sizeof(A_d[0]));
    cudaMalloc((void **)&B_d, n2 * sizeof(B_d[0]));
    cudaMalloc((void **)&C_d, n2 * sizeof(C_d[0]));
     /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(A_h[0]), A_h, 1, A_d, 1);
    status = cublasSetVector(n2, sizeof(B_h[0]), B_h, 1, B_d, 1);
    status = cublasSetVector(n2, sizeof(C_h[0]), C_h, 1, C_d, 1);
    simple_sgemm(N, alpha, A_h, B_h, beta, C_h);
    C_h_ref = C_h;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A_d, N, B_d, N, &beta, C_d, N);
    C_h = (float *)malloc(n2 * sizeof(C_h[0]));
    status = cublasGetVector(n2, sizeof(C_h[0]), C_d, 1, C_h, 1);
    error_norm = 0;
    ref_norm = 0;
    for (i = 0; i < n2; ++i)
    {
        diff = C_h_ref[i] - C_h[i];
        error_norm += diff * diff;
        ref_norm += C_h_ref[i] * C_h_ref[i];
    }
    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    printf("Results:\n");
    printf("- Error norm: %e\n", error_norm);
    printf("- Reference norm: %e\n", ref_norm);
    printf("- Relative error: %e\n", error_norm / ref_norm);
    
    if (error_norm / ref_norm < 1.e-6f) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED\n");
    }

    printf("Cleaning up resources...\n");
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_h_ref);

    cudaFree(A_d);
    cudaFree(B_d);

    cudaFree(C_d);

    status = cublasDestroy(handle);

}