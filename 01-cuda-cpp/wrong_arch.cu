#include <stdio.h>

__global__ void simpleKernel() {}

int main() {
    simpleKernel<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    printf("Error code: %d\n", err);
    printf("Error name: %s\n", cudaGetErrorName(err));
    printf("Error desc: %s\n", cudaGetErrorString(err));
    return 0;
}
