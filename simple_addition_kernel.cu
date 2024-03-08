#include <iostream>
#include <cuda_runtime.h>

__global__ void add_vectors(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    int size = 1024;  
    int* A, * B, * C;

    A = new int[size];
    B = new int[size];
    C = new int[size];

    for (int i = 0; i < size; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_C, size * sizeof(int));

    cudaMemcpy(d_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    add_vectors<<<num_blocks, block_size>>>(d_A, d_B, d_C, size);

    cudaMemcpy(C, d_C, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
