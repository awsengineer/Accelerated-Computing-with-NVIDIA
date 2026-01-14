#include <stdio.h>

__global__ void showThreadInfo() {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d -> Global index: %d\n",
           blockIdx.x, threadIdx.x, globalIdx);
}

int main() {
    printf("Launching 3 blocks x 4 threads = 8 threads:\n\n");
    showThreadInfo<<<3, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
