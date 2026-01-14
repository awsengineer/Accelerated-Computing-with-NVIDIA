#include <stdio.h>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    // Global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Total threads in entire grid
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 30;
    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Calculate grid size
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;  // Round up

    printf("Launching %d blocks x %d threads = %d total threads\n",
           numBlocks, blockSize, numBlocks * blockSize);

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
