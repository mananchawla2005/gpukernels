#include <cuda_runtime.h>
#include <stdio.h>

// Simple error checking macro for CUDA calls
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

// Matrix multiplication kernel (C = A * B)
__global__ void matrixMulKernel(float *A, float *B, float *C, 
                               int ARows, int ACols, int BCols,
                               int rowOffset, int colOffset, int partialWidth) {
    // Calculate row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y + rowOffset;
    int col = blockIdx.x * blockDim.x + threadIdx.x + colOffset;
    
    // Check if this thread handles a valid matrix element
    if (row < ARows && col < BCols) {
        float sum = 0.0f;
        
        // Compute dot product for this part of the matrix
        for (int k = 0; k < partialWidth; k++) {
            sum += A[row * ACols + k] * B[k * BCols + col];
        }
        
        // Write result back to device memory
        C[row * BCols + col] = sum;
    }
}

// Simulate tensor parallelism by splitting a matrix multiplication across "GPUs"
// For demonstration, we'll use thread blocks as simulated GPUs
void tensorParallelMatrixMul(float *A, float *B, float *C, int dim) {
    // Allocate device memory for matrices
    float *d_A, *d_B, *d_C;
    size_t size = dim * dim * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));
    
    // Copy input matrices from host to device
    CHECK_CUDA(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    
    // Clear output matrix
    CHECK_CUDA(cudaMemset(d_C, 0, size));
    
    // Set CUDA kernel launch parameters
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((dim + blockSize.x - 1) / blockSize.x, 
                 (dim + blockSize.y - 1) / blockSize.y);

    // Number of "simulated GPUs" for tensor parallelism
    const int numGPUs = 2;

    // Split the matrix into chunks along the column dimension
    int colsPerGPU = dim / numGPUs;
    
    // Launch kernels for each "simulated GPU" partition
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        int colStart = gpu * colsPerGPU;
        
        // Call matrix multiplication kernel for this partition
        matrixMulKernel<<<gridSize, blockSize>>>(
            d_A, d_B, d_C, 
            dim, dim, dim,         // Matrix dimensions
            0, colStart, dim       // Starting row, starting column, width of computation
        );
    }
    
    // Wait for all kernels to complete
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

// Helper function to initialize a matrix with random values
void initializeMatrix(float *matrix, int dim) {
    for (int i = 0; i < dim * dim; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Helper function to print a small matrix
void printMatrix(float *matrix, int dim) {
    // Print only a small portion if it's large
    int printDim = (dim > 6) ? 6 : dim;
    
    for (int i = 0; i < printDim; i++) {
        for (int j = 0; j < printDim; j++) {
            printf("%.2f ", matrix[i * dim + j]);
        }
        printf("\n");
    }
    if (dim > printDim) printf("...(matrix truncated)...\n");
}

int main() {
    const int matrixDim = 128;  // Matrix dimensions
    
    // Allocate host memory for matrices
    float *A = (float*)malloc(matrixDim * matrixDim * sizeof(float));
    float *B = (float*)malloc(matrixDim * matrixDim * sizeof(float));
    float *C = (float*)malloc(matrixDim * matrixDim * sizeof(float));
    
    // Initialize input matrices with random values
    initializeMatrix(A, matrixDim);
    initializeMatrix(B, matrixDim);
    
    printf("Performing tensor-parallel matrix multiplication using CUDA...\n");
    
    // Execute tensor parallel matrix multiplication
    tensorParallelMatrixMul(A, B, C, matrixDim);
    
    // Print a small portion of the result matrix
    printf("\nResult matrix (truncated):\n");
    printMatrix(C, matrixDim);
    
    // Free host memory
    free(A);
    free(B);
    free(C);
    
    printf("\nTensor parallelism simulation complete!\n");
    
    return 0;
}