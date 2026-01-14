#include <stdio.h>

// Error checking macro - wraps CUDA calls and exits on failure
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void myKernel(float *data) {
    data[threadIdx.x] = threadIdx.x;
}

int main() {
    float *d_data;

    // ========== STEP 1: Allocate GPU memory ==========
    // This is SYNCHRONOUS - if it fails, we know immediately
    CUDA_CHECK(cudaMalloc(&d_data, 256 * sizeof(float)));

    // ========== STEP 2: Launch kernel ==========
    // This is ASYNCHRONOUS - CPU queues work and continues immediately
    // The GPU will execute this in the background
    myKernel<<<1, 256>>>(d_data);

    // ========== STEP 3: Check for launch errors ==========
    // Did the kernel launch fail? (e.g., invalid block size)
    // Note: This does NOT wait for the kernel to finish
    CUDA_CHECK(cudaGetLastError());

    // ========== STEP 4: Wait and check for execution errors ==========
    // Block CPU until GPU finishes, then check for runtime errors
    // (e.g., illegal memory access inside the kernel)
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel executed successfully!\n");

    // ========== STEP 5: Cleanup ==========
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
