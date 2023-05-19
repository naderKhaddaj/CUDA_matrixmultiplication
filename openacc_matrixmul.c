#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024    // Number of rows in matrix A
#define K 512     // Number of columns in matrix A (and rows in matrix B)
#define N 2048    // Number of columns in matrix B

// Function to perform matrix multiplication
void matrixMul(int* A, int* B, int* C, int m, int n, int k) {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int value = 0;
            for (int x = 0; x < k; ++x) {
                value += A[i * k + x] * B[x * n + j];
            }
            C[i * n + j] = value;
        }
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

    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Perform matrix multiplication
    matrixMul(A, B, C, M, N, K);

    // Stop timer
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;

    // Print the execution time
    printf("Matrix multiplication time: %.2f ms\n", elapsedTime);

    // Print the input matrices and the result matrix
    printf("Matrix A:\n");
    printMatrix(A, M, K);

    printf("\nMatrix B:\n");
    printMatrix(B, K, N);

    printf("\nMatrix C (Result):\n");
    printMatrix(C, M, N);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
