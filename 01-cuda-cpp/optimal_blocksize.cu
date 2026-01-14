#include <stdio.h>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 30;
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    // Let CUDA calculate optimal block size
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, add, 0, 0);
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("Optimal block size: %d\n", blockSize);
    printf("Minimum grid size for full occupancy: %d\n", minGridSize);
    printf("Actual grid size: %d\n", numBlocks);

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(x); cudaFree(y);
    return 0;
}
