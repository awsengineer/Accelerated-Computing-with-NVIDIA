#include <stdio.h>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    int index = threadIdx.x;          // Starting position (0-255)
    int stride = blockDim.x;          // Step size (256)

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

    // 1 block, 256 threads
    add<<<1, 256>>>(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
