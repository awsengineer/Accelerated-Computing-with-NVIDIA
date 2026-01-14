#include <stdio.h>

int main() {
    float *d_ptr;

    // cudaMalloc returns cudaError_t - let's capture and inspect it
    cudaError_t err = cudaMalloc(&d_ptr, 1024 * sizeof(float));

    printf("Return value: %d (cudaSuccess = 0)\n", err);
    printf("Error name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));

    // Now let's trigger an error - try to allocate way too much memory
    printf("\n--- Triggering an error ---\n");
    cudaError_t bad_err = cudaMalloc(&d_ptr, (size_t)1024 * 1024 * 1024 * 1024);  // 1 TB!

    printf("Return value: %d\n", bad_err);
    printf("Error name: %s\n", cudaGetErrorName(bad_err));
    printf("Error description: %s\n", cudaGetErrorString(bad_err));

    cudaFree(d_ptr);
    return 0;
}
