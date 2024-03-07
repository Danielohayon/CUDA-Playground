__global__ void kernel(int* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] *= 2;
}

int main() {
    int size = 1024;
    int* data;
    cudaMallocManaged(&data, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        data[i] = i;
    }

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    kernel<<<num_blocks, block_size>>>(data);

    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(data);

    return 0;
}
