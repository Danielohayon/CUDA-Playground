#include <iostream>
#include <cuda_runtime.h>

__global__ void add_vectors(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    int size = 1024;  // Number of elements in the vectors
    int* A, * B, * C;

    // Allocate memory on the host
    A = new int[size];
    B = new int[size];
    C = new int[size];

    // Initialize vectors A and B
    for (int i = 0; i < size; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Allocate memory on the device
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_C, size * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    add_vectors<<<num_blocks, block_size>>>(d_A, d_B, d_C, size);

    // Copy results from device to host
    cudaMemcpy(C, d_C, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the first few elements of the result
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
