#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

int main() {
    hello_kernel<<<2, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
