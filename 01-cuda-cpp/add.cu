#include <stdio.h>
#include <math.h>

// Kernel function - runs on GPU
__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1<<30;  // 1 billion elements
    float *x, *y;

    // Allocate Unified Memory - accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // Initialize on CPU (simple, but slow for large arrays. (We will improve it later)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on GPU (1 block, 1 thread - intentionally slow!)
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing results
    cudaDeviceSynchronize();

    // Check for errors (using CPU, for now)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
       maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
