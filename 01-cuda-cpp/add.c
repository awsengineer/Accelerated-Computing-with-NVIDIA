#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to add elements of two arrays
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1<<30;  // 1 billion elements

    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));

    // Initialize: x=1, y=2
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add(N, x, y);  // Do the addition

    // Check for errors (all values should be 3.0)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    free(x);
    free(y);
    return 0;
}
