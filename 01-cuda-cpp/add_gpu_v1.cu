#include <stdio.h>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 30;  // 1 billion elements
    float *x, *y;

    // Allocate Unified Memory
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // Initialize arrays (on CPU)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel with 1 block, 1 thread
    add<<<1, 1>>>(N, x, y);
    cudaDeviceSynchronize();

    // Verify result
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
