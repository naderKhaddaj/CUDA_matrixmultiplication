#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024    // Number of rows in matrix A
#define K 512    // Number of columns in matrix A (and rows in matrix B)
#define N 2048    // Number of columns in matrix B
#define TILE_SIZE 32   // Tile size for tiling optimization

// CUDA kernel to perform matrix multiplication with tiling
_global_ void matrixMul(int* A, int* B, int* C, int m, int n, int k) {
    // Shared memory for tiles of matrix A and B
    _shared_ int tile_A[TILE_SIZE][TILE_SIZE];
    _shared_ int tile_B[TILE_SIZE][TILE_SIZE];

    // Calculate the row index of the C element and the column index of C element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Accumulator for the result
    int value = 0;

    // Loop over the tiles of matrices A and B
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles of matrix A and B into shared memory
        if (row < m && (t * TILE_SIZE + threadIdx.x) < k)
            tile_A[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if ((t * TILE_SIZE + threadIdx.y) < k && col < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        // Synchronize to ensure tiles are loaded into shared memory
        __syncthreads();

        // Perform matrix multiplication using the loaded tiles
        for (int i = 0; i < TILE_SIZE; ++i)
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        // Synchronize to ensure all threads finish using the tiles before loading new ones
        __syncthreads();
    }

    // Store the result in matrix C
    if (row < m && col < n)
        C[row * n + col] = value;
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
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);

    // Launch kernel to perform matrix multiplication with tiling
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result matrix from device
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

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