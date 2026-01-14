#include <stdio.h>

__global__ void showThreads() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d -> Global ID: %d\n",
           blockIdx.x, threadIdx.x, globalId);
}

int main() {
    printf("Launching 2 blocks x 4 threads = 8 threads total:\n\n");
    showThreads<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
