#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024  // Number of rows in matrix A
#define K 512  // Number of columns in matrix A (and rows in matrix B)
#define N 2048  // Number of columns in matrix B

// CUDA kernel to perform matrix multiplication
_global_ void matrixMul(int* A, int* B, int* C, int m, int n, int k) {
    // Calculate the row index of the C element and the column index of C element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform matrix multiplication only within valid range
    if (row < m && col < n) {
        int value = 0;
        for (int i = 0; i < k; ++i) {
            value += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = value;
    }
}

// Function to initialize matrices with random values
void initializeMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 100;
    }
}

// Function to print a matrix
void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int* A, *B, *C;  // Host matrices
    int* d_A, *d_B, *d_C;  // Device matrices
    int size_A = M * K * sizeof(int);
    int size_B = K * N * sizeof(int);
    int size_C = M * N * sizeof(int);

    // Allocate host memory
    A = (int*)malloc(size_A);
    B = (int*)malloc(size_B);
    C = (int*)malloc(size_C);

    // Initialize host matrices
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);

    // Launch kernel to perform matrix multiplication
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result matrix from device
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);

    // Print the execution time
    printf("Matrix multiplication time: %.2f ms\n", milliseconds);

    // Print the input matrices and the result matrix
    printf("Matrix A:\n");
    printMatrix(A, M, K);

    printf("\nMatrix B:\n");
    printMatrix(B, K, N);

    printf("\nMatrix C (Result):\n");
    printMatrix(C, M, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
