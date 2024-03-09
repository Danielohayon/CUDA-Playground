#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(int* A, int* B, int* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (row < width && col < width) {
        int sum = 0;
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
            printf("Block  x:%d,y:%d Thread x:%d,y:%d => Adding A[%d] + B[%d] to sum\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,row * width + i, i * width + col);
        }
        printf("Block  x:%d,y:%d Thread x:%d,y:%d => Saving sum to C[%d]\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,row * width + col);
        C[row * width + col] = sum;
    }
}

int main() {
    int width = 4;
    int size = width * width;

    int* h_A = new int[size];
    int* h_B = new int[size];
    int* h_C = new int[size];

    for (int i = 0; i < size; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_C, size * sizeof(int));

    cudaMemcpy(d_A, h_A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(2, 2);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (width + block_dim.y - 1) / block_dim.y);

    matrix_multiply<<<grid_dim, block_dim>>>(d_A, d_B, d_C, width);

    cudaMemcpy(h_C, d_C, size * sizeof(int), cudaMemcpyDeviceToHost);
    int m = 0;
    if (10 < width)
      m = 10;
    else
      m = width;
   
    printf("\nA\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << h_A[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    printf("\nB\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << h_B[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    printf("\nC\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << h_C[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
