#include <stdio.h>
#include <math.h>

// Kernel to initialize arrays directly on GPU
__global__ void init(int n, float *x, float *y, float x_val, float y_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x_val;
        y[i] = y_val;
    }
}

__global__ void add(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 30;
    size_t size = N * sizeof(float);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate GPU memory only - no CPU arrays needed!
    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Initialize directly on GPU - no CPU->GPU transfer!
    init<<<numBlocks, blockSize>>>(N, d_x, d_y, 1.0f, 2.0f);

    // Compute
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    // Only copy back what we need to verify
    float *h_y = (float*)malloc(size);
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(h_y);
    return 0;
}
