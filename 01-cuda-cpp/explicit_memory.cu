#include <stdio.h>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 30;  // Using smaller size for this demo
    size_t size = N * sizeof(float);

    // Step 1: Allocate host (CPU) memory
    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);

    // Step 2: Initialize on host
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // Step 3: Allocate device (GPU) memory
    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Step 4: Copy data from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Step 5: Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    // Step 6: Copy results back to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Verify
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    // Step 7: Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}
