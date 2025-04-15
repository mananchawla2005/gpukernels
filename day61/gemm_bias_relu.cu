#define TILE_WIDTH 32
#define COARSENING_FACTOR 4

__global__ void gemm_bias_relu_kernel_coarsened(
    const float* A, const float* W, const float* b, float* C, 
    size_t B, size_t N, size_t M) {
    
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ws[TILE_WIDTH][TILE_WIDTH * COARSENING_FACTOR];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int colStart = (bx * TILE_WIDTH + tx) * COARSENING_FACTOR;

    float sum[COARSENING_FACTOR] = {0};

    for (int p = 0; p < (N + TILE_WIDTH - 1) / TILE_WIDTH; p++) {
        int tiledCol = p * TILE_WIDTH + tx;
        int tiledRow = p * TILE_WIDTH + ty;

        if (row < B && tiledCol < N) {
            As[ty][tx] = A[row * N + tiledCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < COARSENING_FACTOR; ++i) {
            int col = colStart + i;
            if (col < M && tiledRow < N) {
                Ws[ty][tx * COARSENING_FACTOR + i] = W[col * N + tiledRow];
            } else {
                Ws[ty][tx * COARSENING_FACTOR + i] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            float a_val = As[ty][k];
            #pragma unroll
            for (int i = 0; i < COARSENING_FACTOR; ++i) {
                sum[i] += a_val * Ws[k][tx * COARSENING_FACTOR + i];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < COARSENING_FACTOR; ++i) {
        int col = colStart + i;
        if (row < B && col < M) {
            float val = sum[i] + b[col];
            C[row * M + col] = (val > 0.0f) ? val : 0.0f;
        }
    }
}

extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M) {
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((M + TILE_WIDTH * COARSENING_FACTOR - 1) / (TILE_WIDTH * COARSENING_FACTOR),
                  (B + TILE_WIDTH - 1) / TILE_WIDTH);

    gemm_bias_relu_kernel_coarsened<<<gridSize, blockSize>>>(A, W, b, C, B, N, M);
}