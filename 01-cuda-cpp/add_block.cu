#include <stdio.h>
#include <math.h>

__global__
void add(int n, float *x, float *y) {
    int index = threadIdx.x;      // This thread's starting index
    int stride = blockDim.x;      // Total threads = step size
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1<<30;  // 1 billion elements
    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 1 block, 256 threads (better, but still limited)
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
