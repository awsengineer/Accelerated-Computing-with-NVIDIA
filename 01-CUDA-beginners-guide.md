# CUDA C/C++ - A Beginner's Guide

Learn GPU programming from the ground up. By the end of this guide, you'll understand how to write parallel code that runs on NVIDIA GPUs.

**What you'll learn:**
1. Why GPUs are fast (and when they're not)
2. The CPU-GPU programming model
3. Writing and launching GPU kernels
4. Thread organization: threads, blocks, and grids
5. Memory management between CPU and GPU
6. Profiling and debugging CUDA code

**Prerequisites:** Basic C programming knowledge. Setup instructions are in [Appendix A](#appendix-a-setup).

*Inspired by Mark Harris's [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/).*

---
## 1. Why GPU Programming?

Consider adding two arrays of 1 billion numbers:

```
x = [1, 1, 1, ...] (1 billion elements)
y = [2, 2, 2, ...] (1 billion elements)
result: y = [3, 3, 3, ...]
```

On a CPU, you'd write a loop that processes elements one by one:

```c
for (int i = 0; i < 1000000000; i++)
    y[i] = x[i] + y[i];
```

This takes about 15-20 seconds (depends on the CPU obviously). A single CPU core handles only a few elements at a time; even with vectorization and several cores you have on the order of tens of lanes, not the tens of thousands a GPU offers.

**The insight:** Each addition is independent. Element 0 doesn't need element 1's result. What if we could do all 1 billion additions *at the same time*?

That's what GPUs help to do.

---
## 2. CPU vs GPU: The Mental Model

| | CPU | GPU |
|---|---|---|
| **Design philosophy** | Few fast cores | Many slower cores |
| **Core count** | 4-64 cores | 1,000-16,000 cores |
| **Optimized for** | Complex sequential tasks | Simple parallel tasks |
| **Memory** | System RAM ("host memory") | VRAM ("device memory") |
| **Code terminology** | Host code | Device code / Kernels |

**Key insight:** GPUs are fast because they do the *same operation* on *many data points* simultaneously. This is called **data parallelism**.

> **What a "core" means here — don't be misled by the count.** A CPU core and a GPU "CUDA core" are not the same kind of thing. A CPU core is a full, independent processor (its own instruction stream, branch prediction, out-of-order execution). A **CUDA core is just one arithmetic lane** — it can't fetch or decode instructions on its own. The real independent processing units on a GPU are its **Streaming Multiprocessors (SMs)**: this notebook's NVIDIA L4 has **58 SMs**, each containing 128 CUDA cores (7,424 total). So "16,000 cores" doesn't mean 16,000 independent programs — it means a few dozen SMs, each driving many lanes in lockstep. We'll unpack how those lanes execute (in groups of 32, called *warps*) in Section 4.

### When GPUs Help (and When They Don't)

**Good for GPUs:**
- Array/matrix operations (same operation on millions of elements)
- Image processing (same filter applied to millions of pixels)
- Neural network inference (matrix multiplications)
- Physics simulations (same equations for many particles)

**Bad for GPUs:**
- Sequential algorithms where step N depends on step N-1
- Workloads with heavy branching (if/else) that differs per element
- Small datasets (overhead exceeds benefit)
- Tasks requiring lots of CPU-GPU communication

### Your Instance and GPU

Let's see what instance type and GPU you're working with:


```bash
%%bash
# Get instance type via IMDSv2
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null)
INSTANCE_TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null)

echo "════════════════════════════════════════════════════════════════"
echo "                      INSTANCE DETAILS"
echo "════════════════════════════════════════════════════════════════"
if [ -n "$INSTANCE_TYPE" ]; then
    echo "Instance Type : $INSTANCE_TYPE"
    echo "Region        : $REGION"
    echo ""
    echo "Expected GPU (from AWS):"
    aws ec2 describe-instance-types --instance-types $INSTANCE_TYPE --region $REGION \
        --query 'InstanceTypes[0].GpuInfo.Gpus[0].[Name, Manufacturer, Count, MemoryInfo.SizeInMiB]' \
        --output text 2>/dev/null | awk '{printf "  GPU         : %s %s\n  Count       : %s\n  VRAM        : %s MiB\n", $2, $1, $3, $4}' \
        || echo "  No GPU configured for this instance type"
else
    echo "Not running on EC2"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                      DETECTED GPU (by nvidia-smi command)"
echo "════════════════════════════════════════════════════════════════"
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: No GPU detected!"
    echo "   This notebook requires an NVIDIA GPU."
    echo "   Use a GPU instance (g4dn.xlarge, p3.2xlarge, etc.)"
else
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | \
        awk -F', ' '{printf "  GPU         : %s\n  VRAM        : %s\n  Compute Cap : %s\n", $1, $2, $3}'
fi
echo "════════════════════════════════════════════════════════════════"
```

    ════════════════════════════════════════════════════════════════
                          INSTANCE DETAILS
    ════════════════════════════════════════════════════════════════
    Instance Type : g6.xlarge
    Region        : us-east-1
    
    Expected GPU (from AWS):
    
    ════════════════════════════════════════════════════════════════
                          DETECTED GPU (by nvidia-smi command)
    ════════════════════════════════════════════════════════════════
      GPU         : NVIDIA L4
      VRAM        : 23034 MiB
      Compute Cap : 8.9
    ════════════════════════════════════════════════════════════════


**Understanding the output:**

| Field | Example | Meaning |
|-------|---------|--------|
| name | NVIDIA L4 | GPU model |
| memory.total | 23034 MiB | VRAM available (~23 GB) |
| compute_cap | 8.9 | Architecture version (for compiler flags) |

The **compute capability** tells you which features your GPU supports and which compiler flag to use:

| Compute Capability | Architecture | Compiler Flag |
|-------------------|--------------|---------------|
| 7.5 | Turing (T4, RTX 20xx) | `-arch=sm_75` |
| 8.0 | Ampere (A100) | `-arch=sm_80` |
| 8.6 | Ampere (RTX 30xx) | `-arch=sm_86` |
| 8.9 | Ada (RTX 40xx) | `-arch=sm_89` |
| 9.0 | Hopper (H100) | `-arch=sm_90` |

**Important:** Match your compile flag to your GPU:
- Compiling for a **higher** architecture (e.g., `-arch=sm_80` on a sm_75 GPU) will fail with `cudaErrorNoKernelImageForDevice`
- Compiling for a **lower** architecture works but may miss optimizations for your GPU
- When in doubt, use `-arch=native` (CUDA 11.5+) to auto-detect your GPU

Let's see what happens when we compile for the wrong architecture (we will learn error handling in a later section):


```python
%%writefile wrong_arch.cu
#include <stdio.h>

__global__ void simpleKernel() {}

int main() {
    simpleKernel<<<1, 1>>>();
    
    cudaError_t err = cudaGetLastError();
    printf("Error code: %d\n", err);
    printf("Error name: %s\n", cudaGetErrorName(err));
    printf("Error desc: %s\n", cudaGetErrorString(err));
    return 0;
}
```

    Overwriting wrong_arch.cu



```bash
%%bash
# Compile for sm_90 (Hopper) but run on L4 (sm_89) - this will fail!
/usr/local/cuda/bin/nvcc -arch=sm_90 wrong_arch.cu -o wrong_arch && ./wrong_arch
```

    Error code: 209
    Error name: cudaErrorNoKernelImageForDevice
    Error desc: no kernel image is available for execution on the device


---
## 3. Your First CUDA Program

Let's start with a CPU program that adds two arrays, then convert it to CUDA.

### The CPU Version


```python
%%writefile add_cpu.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 30;  // 1 billion elements (1<<30 = 2^30)
    
    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    add(N, x, y);  // Add arrays
    
    // Verify result (all elements should be 3.0)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);
    
    free(x);
    free(y);
    return 0;
}
```

    Overwriting add_cpu.c



```bash
%%bash
gcc add_cpu.c -o add_cpu -lm && time ./add_cpu
```

    Max error: 0.000000


    
    real	0m12.695s
    user	0m9.309s
    sys	0m3.378s


Took 12 seconds on EC2 Instance.

### Converting to CUDA: Three Changes

To run this on a GPU, we need  **three changes**:

#### Change 1: Mark the function with `__global__`

```c
// CPU version
void add(int n, float *x, float *y) { ... }

// GPU version
__global__ void add(int n, float *x, float *y) { ... }
```

The `__global__` keyword tells the compiler: "This function runs on the GPU but is called from the CPU."

Functions marked `__global__` are called **kernels**.

#### Change 2: Use CUDA memory allocation

```c
// CPU version
float *x = malloc(N * sizeof(float));
free(x);

// GPU version (Unified Memory)
float *x;
cudaMallocManaged(&x, N * sizeof(float));
cudaFree(x);
```

`cudaMallocManaged` allocates **Unified Memory** - memory accessible from both CPU and GPU. The CUDA runtime automatically handles data movement.

#### Change 3: Launch with execution configuration

```c
// CPU version
add(N, x, y);

// GPU version
add<<<1, 1>>>(N, x, y);   // Launch kernel
cudaDeviceSynchronize();   // Wait for GPU to finish
```

The `<<<blocks, threads>>>` syntax specifies how many parallel threads to launch. We'll explore this in detail soon.

`cudaDeviceSynchronize()` makes the CPU wait for the GPU to finish - kernel launches are *asynchronous* (the CPU continues immediately).

### The CUDA Version


```python
%%writefile add_gpu_v1.cu
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
```

    Overwriting add_gpu_v1.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 add_gpu_v1.cu -o add_gpu_v1 && time ./add_gpu_v1
```

    Max error: 0.000000


    
    real	1m52.451s
    user	1m48.567s
    sys	0m3.890s


**It works!** But it's actually *slower* than the CPU version! Why? We're only using 1 GPU thread.

To make it fast, we need to understand GPU threads.

---
## 4. GPU Thread Organization

GPUs organize threads into a hierarchy:

```
Grid (all threads for one kernel launch)
└── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ... (up to 1024 threads)
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
└── ... (thousands of blocks)
```

### Why Two Levels?

**Threads within a block** can:
- Share fast on-chip memory (shared memory)
- Synchronize with each other
- Cooperate on a task

**Threads in different blocks** cannot:
- Share memory directly
- Synchronize (they may run at different times)

This design allows the GPU to schedule blocks independently across its processors.

### Built-in Thread Variables

Every thread can identify itself using built-in variables:

| Variable | Meaning | Example |
|----------|---------|--------|
| `threadIdx.x` | Thread index within block | "I'm thread 5 in my block" |
| `blockIdx.x` | Block index within grid | "I'm in block 2" |
| `blockDim.x` | Threads per block | "My block has 256 threads" |
| `gridDim.x` | Blocks in grid | "The grid has 4096 blocks" |

### The Global Index Formula

To get a unique index for each thread across the entire grid:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**Example:** Block 2, Thread 5, with 256 threads per block:
```
i = 2 * 256 + 5 = 517
```

This thread processes `array[517]`.

### Interactive visualization
Before diving into the details, explore how threads, blocks, and the grid fit together in this interactive diagram:
**[Understanding Grid, Block, and Thread](https://htmlpreview.github.io/?https://github.com/awsengineer/Accelerated-Computing-with-NVIDIA/blob/main/01-cuda-cpp/01_cuda-grid-block-thread.html)**

### Now run it:


```python
%%writefile show_threads.cu
#include <stdio.h>

__global__ void showThreadInfo() {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d -> Global index: %d\n",
           blockIdx.x, threadIdx.x, globalIdx);
}

int main() {
    printf("Launching 3 blocks x 4 threads = 12 threads:\n\n");
    showThreadInfo<<<3, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

    Overwriting show_threads.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 show_threads.cu -o show_threads && ./show_threads
```

    Launching 3 blocks x 4 threads = 12 threads:
    
    Block 1, Thread 0 -> Global index: 4
    Block 1, Thread 1 -> Global index: 5
    Block 1, Thread 2 -> Global index: 6
    Block 1, Thread 3 -> Global index: 7
    Block 0, Thread 0 -> Global index: 0
    Block 0, Thread 1 -> Global index: 1
    Block 0, Thread 2 -> Global index: 2
    Block 0, Thread 3 -> Global index: 3
    Block 2, Thread 0 -> Global index: 8
    Block 2, Thread 1 -> Global index: 9
    Block 2, Thread 2 -> Global index: 10
    Block 2, Thread 3 -> Global index: 11


**Notice:** The output order is unpredictable! Threads run in parallel, not sequentially. Never assume execution order.

### Why `.x`?

CUDA supports 1D, 2D, and 3D thread layouts. For arrays, 1D (`.x` only) is sufficient. For images, you might use 2D (`.x` and `.y`). For volumes, 3D.

```c
// 2D example for image processing
// Remember, indexes start with 0.
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### Warps and Memory Coalescing

Two facts about how the hardware *actually* runs your threads explain most CUDA performance surprises. Neither changes what your code computes — only how fast it runs.

**1. Threads run in warps of 32.** Inside a block, the GPU does not schedule threads one by one. It groups them into **warps** of 32 consecutive threads (`threadIdx.x` 0–31, 32–63, …) and executes them in **lockstep**: at each step, all 32 lanes run the *same instruction* on their own data. This is NVIDIA's **SIMT** (Single Instruction, Multiple Threads) model. Two consequences:

- It's why **block size should be a multiple of 32** (Section 5). A block of 100 threads still occupies 4 warps = 128 lanes; 28 lanes sit idle every instruction.
- **Branch divergence:** if an `if/else` sends some lanes of a warp one way and the rest the other, the hardware runs *both* paths and masks off the inactive lanes each time — so a divergent branch inside a warp costs as much as doing both halves. Branches that are uniform across a warp are free; branches that split a warp are not.

**2. Memory is read in cache-line chunks, so neighbors should touch neighbors.** Global memory is delivered in aligned segments (cache lines), not single floats. When the 32 threads of a warp read **32 consecutive addresses**, the hardware merges them into a few wide transactions — this is **coalescing**, and it's the efficient case. If those same 32 threads read addresses far apart (a strided or column-wise pattern), each lands in a different cache line and the hardware issues many separate transactions, moving far more data than you use.

The rule of thumb: **make consecutive threads (`threadIdx.x`) touch consecutive memory addresses.** The kernel below copies the same matrix two ways — once coalesced, once not — to measure the difference on this GPU.


```python
%%writefile coalescing.cu
#include <stdio.h>

// We copy a ROWS x COLS matrix two ways. BOTH kernels touch every element
// exactly once -- only the thread -> address mapping differs.

// COALESCED: consecutive threads (threadIdx.x) map to consecutive columns,
// which are consecutive addresses in row-major memory. A warp's 32 reads
// fall in the same handful of cache lines -> merged into few transactions.
__global__ void rowmajor(const float *in, float *out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (c < cols && r < rows) { int i = r * cols + c; out[i] = in[i]; }
}

// UNCOALESCED: consecutive threads map to consecutive ROWS, whose addresses
// are `cols` apart. Each lane of the warp lands in a different cache line
// -> many separate transactions for the same 32 values.
__global__ void colmajor(const float *in, float *out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (c < cols && r < rows) { int i = r * cols + c; out[i] = in[i]; }
}

float time_kernel(void(*k)(const float*,float*,int,int),
                  const float*in,float*out,int rows,int cols,dim3 grid,dim3 block){
    k<<<grid,block>>>(in,out,rows,cols); cudaDeviceSynchronize();   // warmup
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); k<<<grid,block>>>(in,out,rows,cols); cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e);
    cudaEventDestroy(s); cudaEventDestroy(e); return ms;
}

int main() {
    int rows = 8192, cols = 8192;
    long n = (long)rows * cols;
    size_t bytes = n * sizeof(float);
    int dev = 0; cudaGetDevice(&dev);

    float *in, *out;
    cudaMallocManaged(&in, bytes);
    cudaMallocManaged(&out, bytes);
    for (long i = 0; i < n; i++) { in[i] = 1.0f; out[i] = 0.0f; }

    // Prefetch to the GPU so we time the kernels, not page migration.
    cudaMemLocation loc; loc.type = cudaMemLocationTypeDevice; loc.id = dev;
    cudaMemPrefetchAsync(in, bytes, loc, 0, 0);
    cudaMemPrefetchAsync(out, bytes, loc, 0, 0);
    cudaDeviceSynchronize();

    dim3 block(256);
    dim3 gridR((cols + 255) / 256, rows);   // x over columns -> coalesced
    dim3 gridC((rows + 255) / 256, cols);   // x over rows    -> uncoalesced

    float rm = time_kernel(rowmajor, in, out, rows, cols, gridR, block);
    float cm = time_kernel(colmajor, in, out, rows, cols, gridC, block);

    printf("%dx%d matrix, all %ld elements copied each way:\n", rows, cols, n);
    printf("  coalesced   (adjacent threads -> adjacent addresses): %.3f ms\n", rm);
    printf("  uncoalesced (adjacent threads -> %d apart)          : %.3f ms   (%.1fx slower)\n",
           cols, cm, cm / rm);

    cudaFree(in); cudaFree(out);
    return 0;
}
```

    Writing coalescing.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 coalescing.cu -o coalescing && ./coalescing
```

    8192x8192 matrix, all 67108864 elements copied each way:
      coalesced   (adjacent threads -> adjacent addresses): 2.266 ms
      uncoalesced (adjacent threads -> 8192 apart)          : 4.828 ms   (2.1x slower)


---
## 5. Making It Parallel

Our first CUDA program used `<<<1, 1>>>` - one thread doing all the work. Let's fix that.

### Version 2: One Block, Many Threads

With 256 threads, each thread handles every 256th element:

```
Thread 0: elements 0, 256, 512, 768, ...
Thread 1: elements 1, 257, 513, 769, ...
Thread 2: elements 2, 258, 514, 770, ...
```

This is called a **stride loop**:


```python
%%writefile add_gpu_v2.cu
#include <stdio.h>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    //threadIdx.x is the starting position (0-255)
    //blockDim.x is the step size (256)
    
    for (int i = threadIdx.x; i < n; i += blockDim.x)
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
```

    Overwriting add_gpu_v2.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 add_gpu_v2.cu -o add_gpu_v2 && time ./add_gpu_v2
```

    Max error: 0.000000


    
    real	0m26.770s
    user	0m22.836s
    sys	0m3.919s


Better! But GPUs have thousands of cores organized into multiple Streaming Multiprocessors (SMs). One block only runs on one SM. We need more blocks.

### Review a few concepts:
- "CUDA core" = Marketing term for one FP32 Arithmetic Logic Unit (ALU) (formerly knows as a "shader" processor in the video and gaming era). Far simpler than a CPU core: no out-of-order execution or branch prediction, because fetch/decode/scheduling are shared across the warp at the warp-scheduler level inside the SM (an SM has several warp schedulers).
  
- The SM is the architecturally meaningful unit for tuning: it's what blocks are scheduled onto and the granularity where occupancy and register/shared-memory limits apply.

- Hierarchy: GPU → SMs → CUDA cores. An SM is a compute cluster holding many cores. Total cores = SMs × cores-per-SM.

- g6.xlarge instance: NVIDIA L4, sm_89, 128 cores/SM, warp 32, CUDA 13.1.

- SM is the tuning unit — blocks schedule onto it; occupancy and register/shared-memory limits apply there.

### Version 3: Many Blocks, Many Threads (Full Parallelization)

**How the kernel splits the work — the grid-stride loop**

The GPU runs **one copy of the kernel per thread**. Every copy runs the same code; only the built-in variables (`blockIdx`, `threadIdx`, …) differ. So each thread's only job is: *work out which elements are mine, then do those.*

Two values set that up:

| Variable | Formula | Meaning |
|----------|---------|---------|
| `index` | `blockIdx.x * blockDim.x + threadIdx.x` | this thread's unique number across the grid → where it **starts** |
| `stride` | `blockDim.x * gridDim.x` | total threads launched → how far it **jumps** each pass |

Each thread starts at its own `index` and hops forward by `stride` until it runs past the end. With **4 threads** over a **10-element** array (`stride = 4`):

```
thread 0 → 0, 4, 8       thread 2 → 2, 6
thread 1 → 1, 5, 9       thread 3 → 3, 7
```

**Why write it this way?** The same code works for *any* array size:
- **More threads than elements** → each thread's loop body runs once, then stops.
- **More elements than threads** → each thread loops again to pick up the extras.


```python
%%writefile add_gpu_v3.cu
#include <stdio.h>
#include <math.h>

// Grid-stride loop (see the cell above for how the work is split).
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's start position
    int stride = blockDim.x * gridDim.x;                // total threads = the jump size

    for (int i = index; i < n; i += stride)             // "i < n" also keeps us in bounds
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 25;                 // ~33M elements (128 MB/array) - small enough to time cleanly
    size_t bytes = (size_t)N * sizeof(float);
    float *x, *y;

    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;  // = ceil(N / blockSize)

    printf("Launching %d blocks x %d threads = %d total threads\n",
           numBlocks, blockSize, numBlocks * blockSize);

    // --- IMPORTANT: prefetch managed memory to the GPU BEFORE timing ---
    // The arrays were just written on the CPU, so their pages live in host
    // memory. On first GPU access they page-fault and migrate device-ward; if
    // we don't move them first, that migration time lands INSIDE our timer and
    // we'd be measuring data movement, not the kernel. (We can't simply do a
    // warm-up launch here the way the matrix-multiply cells do, because this
    // kernel is not idempotent: running y = x + y twice gives 4, not 3.)
    int device = 0;
    cudaGetDevice(&device);
    cudaMemLocation loc;                       // CUDA 13 prefetch takes a location struct
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = device;
    cudaMemPrefetchAsync(x, bytes, loc, 0, 0);
    cudaMemPrefetchAsync(y, bytes, loc, 0, 0);
    cudaDeviceSynchronize();

    // --- Time ONLY the kernel using CUDA events (GPU-timeline timers) ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);              // wait until the kernel + stop event complete
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);  // elapsed GPU time between the events, in ms
    printf("Kernel time (blockSize=%d): %.3f ms\n", blockSize, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
```

    Overwriting add_gpu_v3.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 add_gpu_v3.cu -o add_gpu_v3 &&  time ./add_gpu_v3
```

    Launching 131072 blocks x 256 threads = 33554432 total threads
    Kernel time (blockSize=256): 1.701 ms
    Max error: 0.000000


    
    real	0m0.569s
    user	0m0.273s
    sys	0m0.274s


### Choosing Optimal Block Size

We used 256 threads per block, but is that optimal? CUDA provides a way to calculate the best block size for your kernel.

**Key factors:**
- Block size must be a multiple of 32 (warp size)
- Maximum is 1024 threads per block
- Optimal size depends on kernel's register and shared memory usage

Use `cudaOccupancyMaxPotentialBlockSize` to let CUDA calculate it:


```python
%%writefile optimal_blocksize.cu
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
    
    // For my kernel add, what block size gives the best occupancy on this GPU?
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
```

    Overwriting optimal_blocksize.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 optimal_blocksize.cu -o optimal_blocksize && time ./optimal_blocksize
```

    Optimal block size: 768
    Minimum grid size for full occupancy: 116
    Actual grid size: 1398102
    Max error: 0.000000


    
    real	0m25.447s
    user	0m21.593s
    sys	0m3.835s


For simple kernels like ours, 256 or 1024 are typically optimal. Complex kernels using more registers or shared memory may need smaller block sizes.

### Version 4: Running with the Optimal Block Size

In the previous section, `cudaOccupancyMaxPotentialBlockSize` reported **768** as the optimal block size for our `add` kernel on this GPU (an NVIDIA L4). Let's run the kernel with that value hardcoded — `blockSize = 768` instead of the default 256 — and time it.

The kernel itself is unchanged from Version 3; only the launch configuration differs. For a simple, memory-bound kernel like this one, don't expect a dramatic speedup: the workload is limited by memory bandwidth, not by how many threads each block holds. The point is to show how you'd *apply* the recommended block size in practice.


```python
%%writefile add_gpu_v4.cu
#include <stdio.h>
#include <math.h>

// Same grid-stride kernel as v3 - only the launch block size changes.
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 25;                 // same size as v3, for a fair comparison
    size_t bytes = (size_t)N * sizeof(float);
    float *x, *y;
    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);

    for (int i = 0; i < N; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    // Use the optimal block size reported by cudaOccupancyMaxPotentialBlockSize
    // in the previous section (768 on this GPU), instead of the default 256.
    int blockSize = 768;
    int numBlocks = (N + blockSize - 1) / blockSize;  // = ceil(N / blockSize)

    printf("Launching %d blocks x %d threads = %d total threads\n",
           numBlocks, blockSize, numBlocks * blockSize);

    // Prefetch managed memory to the GPU first so the timer measures the
    // kernel, not page migration (see v3 for the full explanation).
    int device = 0;
    cudaGetDevice(&device);
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = device;
    cudaMemPrefetchAsync(x, bytes, loc, 0, 0);
    cudaMemPrefetchAsync(y, bytes, loc, 0, 0);
    cudaDeviceSynchronize();

    // --- Time ONLY the kernel using CUDA events (same method as v3) ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);              // wait until the kernel + stop event complete
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);  // elapsed GPU time between the events, in ms
    printf("Kernel time (blockSize=%d): %.3f ms\n", blockSize, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
```

    Overwriting add_gpu_v4.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 add_gpu_v4.cu -o add_gpu_v4 && time ./add_gpu_v4
```

    Launching 43691 blocks x 768 threads = 33554688 total threads
    Kernel time (blockSize=768): 1.677 ms
    Max error: 0.000000


    
    real	0m0.572s
    user	0m0.280s
    sys	0m0.269s


### Why the optimal block size barely changed the time

Compare the **`Kernel time`** lines from Version 3 (256 threads) and Version 4 (768 threads): they're nearly identical — a difference of only a few percent — even though 768 was the "optimal" block size. Two things explain this, and both matter.

**1. Measure the kernel, not the whole program.**
The wall-clock `time` of the program is dominated by overhead that has nothing to do with the kernel: allocating memory, the CPU loop that fills the arrays, page migration, and the CPU verify loop. The kernel itself is a tiny slice. That's why we wrap *just* the kernel launch in **CUDA events** (`cudaEventRecord` / `cudaEventElapsedTime`), and why we **prefetch the managed arrays to the GPU first** — otherwise the first GPU access would page-fault and migrate data *inside* our timing window, and we'd be measuring data movement instead of compute. The `Kernel time` line is the number to compare, not `real`.

**2. Even the clean kernel time barely moves — because this kernel is memory-bound.**
The `add` kernel does almost no arithmetic (one addition per element) but moves a lot of data: it reads `x`, reads `y`, and writes `y`. The bottleneck is **memory bandwidth**, not the compute units. Block size controls how threads are *packed onto* the SMs (occupancy); it does nothing to make VRAM stream data faster. At 256 threads the kernel is already close to saturating bandwidth, so moving to 768 gives only a small improvement.

> **The lesson:** `cudaOccupancyMaxPotentialBlockSize` tells you the block size with the *best occupancy*, but **occupancy only helps when occupancy is the limiter.** For a memory-bandwidth-bound kernel, you're already near the hardware ceiling, so tuning the block size can't buy much.

So when *does* block size matter more? When a kernel does enough arithmetic per byte that it stops being limited by memory bandwidth. The next section builds a matrix multiply — far more math per byte loaded — and we'll see the block size make a much larger difference there.

## 6. A Real Workload: Matrix Multiplication

Array addition was a weak demonstration of GPU power. As we saw, it is **memory-bandwidth-bound**: it does almost no arithmetic (one add per element) while moving a lot of data, so the arithmetic units sit mostly idle and block size barely matters.

Matrix multiplication (GEMM) is a far better workload. Its key property is **high arithmetic intensity** — it does O(N³) math on only O(N²) data, so each value loaded from memory can be *reused* many times instead of used once. That reuse is what lets a kernel keep the arithmetic units busy, and it's why GEMM is the core operation behind 3D graphics, scientific computing, and every neural network. When people say GPUs power modern AI, this is the operation they mean — though modern GPUs run it on dedicated **Tensor Cores** in lower precision (TF32/FP16/FP8), not the plain FP32 CUDA-core version we write here.

A heads-up on honesty, since we'll measure as we go: the simple kernels in this section do **not** actually reach the GPU's compute limit. They run at only a few percent of the L4's ~30 TFLOP/s FP32 peak, because naive GEMM still re-reads the same data from global memory over and over — so it stays limited by memory bandwidth and cache, not arithmetic. The point of this section is to show *why* GEMM has the potential to be compute-bound, and how the central optimization (shared-memory **tiling**) chips away at the memory bottleneck to get closer.

In this section we will:
1. Write a naive GPU matrix multiply and time *only the kernel*.
2. See that block size now makes a real difference (unlike with add).
3. Cut memory traffic with **shared-memory tiling**.
4. Compare against a CPU to see the scale of the difference.

Throughout, we initialize the matrices **on the GPU** (no CPU fill loop, no host-to-device copy) so the timings reflect compute, not data movement.

### What matrix multiplication computes

For two N x N matrices A and B, the product C = A x B is defined element by element:

```
C[row][col] = sum over k of  A[row][k] * B[k][col]
```

Each output element is the **dot product** of one row of A and one column of B — that is N multiply-add operations. With N x N output elements, the whole multiply costs about **N^3 multiply-adds** while the data is only about **N^2 numbers**.

That ratio is the key. The amount of math (N^3) grows much faster than the amount of data (N^2), so in principle each value loaded from memory can be **reused** many times. That high *arithmetic intensity* is what gives GEMM the *potential* to be compute-bound — unlike array addition, where every value is used exactly once. (As we'll measure, the naive kernel doesn't realize that potential, because it re-reads from slow global memory instead of reusing from fast on-chip memory — that's what tiling fixes later.)

**The parallel mapping is natural:** assign **one thread to each output element** `C[row][col]`. This is exactly the 2D thread indexing introduced in Section 4:

```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Each thread then loops over `k` to compute its own dot product. Threads are laid out in a 2D grid of 2D blocks, matching the 2D shape of the output matrix.

**Checking the result (gold sample):** we fill A and B entirely with `1.0`. Then every dot product is a sum of N ones, so *every* element of C must equal exactly **N** — an easy correctness check. (One caveat: all-ones can't catch an indexing bug, since any wrong-but-still-N-terms sum also gives N. The tiled cell adds a stronger check against the naive result with varied data.)


```python
%%writefile gemm_naive.cu
#include <stdio.h>
#include <math.h>

// Initialize a matrix to a constant value, on the GPU.
// (No CPU fill loop and no host-to-device copy.)
__global__ void fill(float *m, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] = val;
}

// Naive GEMM: one thread computes one output element C[row][col].
// It reads a full row of A and a full column of B from global memory.
__global__ void gemm(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 2D indexing (Section 4)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++)
            acc += A[row * N + k] * B[k * N + col];   // dot product of row & col
        C[row * N + col] = acc;
    }
}

int main() {
    int N = 10240;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Fill A = B = 1.0 ON THE GPU -> every C element must come out equal to N.
    int total = N * N, bs = 256, nb = (total + bs - 1) / bs;
    fill<<<nb, bs>>>(A, total, 1.0f);
    fill<<<nb, bs>>>(B, total, 1.0f);
    cudaDeviceSynchronize();

    // 2D launch: 16x16 threads per block, enough blocks to cover the N x N output.
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warm up once (the first launch pays one-time setup costs), then time the kernel.
    gemm<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gemm<<<grid, block>>>(A, B, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = 2.0 * (double)N * N * N / (ms / 1000.0) / 1e9;
    double peak = 30300.0;   // L4 FP32 peak ~30.3 TFLOP/s = 30300 GFLOP/s
    double pctPeak = 100.0 * gflops / peak;
    printf("Naive GEMM  N=%d  block 16x16\n", N);
    printf("Kernel time: %.3f ms   (%.0f GFLOP/s = %.1f%% of ~30.3 TFLOP/s peak)\n",
           ms, gflops, pctPeak);

    // Gold check: every element should equal N.
    float maxError = 0.0f;
    for (int i = 0; i < total; i++)
        maxError = fmaxf(maxError, fabsf(C[i] - (float)N));
    printf("Max error vs %d: %.1f\n", N, maxError);
    printf("(Only %.1f%% of peak -> this kernel is still memory/cache-bound, not compute-bound.)\n",
           pctPeak);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
```

    Overwriting gemm_naive.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 gemm_naive.cu -o gemm_naive && ./gemm_naive
```

    Naive GEMM  N=10240  block 16x16
    Kernel time: 1932.359 ms   (1111 GFLOP/s = 3.7% of ~30.3 TFLOP/s peak)
    Max error vs 10240: 0.0
    (Only 3.7% of peak -> this kernel is still memory/cache-bound, not compute-bound.)


### Why block size matters now

With array addition, changing the block size barely moved the kernel time — the kernel was bandwidth-limited and occupancy was not the bottleneck. Matrix multiply has much more arithmetic per byte, so keeping enough warps resident per SM (to hide memory and instruction latency) makes a bigger difference. That residency is exactly what block size controls — so here the block-size choice should show up clearly in the timings.

Let's run the **same naive kernel** at several block sizes and compare the kernel times. (Recall block size must be a multiple of 32 and at most 1024 threads; for a 2D block, `tile x tile` threads, the values 8x8=64, 16x16=256, 32x32=1024 are all valid.)


```python
%%writefile gemm_blocksize.cu
#include <stdio.h>
#include <math.h>

__global__ void fill(float *m, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] = val;
}
__global__ void gemm(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) acc += A[row*N+k] * B[k*N+col];
        C[row*N+col] = acc;
    }
}

// Time the kernel once for a given square block size (tile x tile threads).
float time_block(const float *A, const float *B, float *C, int N, int tile) {
    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (N + tile - 1) / tile);
    gemm<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();                 // warmup
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    gemm<<<grid, block>>>(A, B, C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main() {
    int N = 10240;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);
    int total = N * N, bs = 256, nb = (total + bs - 1) / bs;
    fill<<<nb, bs>>>(A, total, 1.0f);
    fill<<<nb, bs>>>(B, total, 1.0f);
    cudaDeviceSynchronize();

    double work = 2.0 * (double)N * N * N / 1e9;   // GFLOP
    printf("Naive GEMM, N=%d  -  effect of block size:\n", N);
    int tiles[3] = {8, 16, 32};
    for (int i = 0; i < 3; i++) {
        int t = tiles[i];
        float ms = time_block(A, B, C, N, t);
        printf("  block %2dx%-2d (%4d threads): %7.3f ms   (%6.0f GFLOP/s)\n",
               t, t, t * t, ms, work / (ms / 1000.0));
    }
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
```

    Overwriting gemm_blocksize.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 gemm_blocksize.cu -o gemm_blocksize && ./gemm_blocksize
```

    Naive GEMM, N=10240  -  effect of block size:
      block  8x8  (  64 threads): 5320.885 ms   (   404 GFLOP/s)
      block 16x16 ( 256 threads): 1949.765 ms   (  1101 GFLOP/s)
      block 32x32 (1024 threads): 1652.409 ms   (  1300 GFLOP/s)


### Going faster: shared memory and tiling

Look again at the naive kernel. To compute one element, a thread reads an entire row of A and an entire column of B **from global memory** (VRAM) — the slow, off-chip memory. Worse, neighboring threads re-read much of the same data. Every value of A and B ends up being fetched from global memory many times.

**Shared memory** fixes this. Recall the memory hierarchy from Section 8: shared memory is a small, very fast, on-chip scratchpad that is **shared by all threads in a block**. The idea, called **tiling**, is:

1. The threads in a block **cooperatively load** a small `TILE x TILE` square of A and of B from global memory into shared memory — each value fetched from global memory just **once**.
2. They call `__syncthreads()` so everyone waits until the tile is fully loaded.
3. They compute using the fast shared-memory copies, reusing each loaded value `TILE` times.
4. `__syncthreads()` again, then slide to the next tile along the `k` dimension and repeat, accumulating the partial sums.

`__syncthreads()` is a **barrier**: every thread in the block waits there until all threads arrive. It is only possible *because* a block runs on a single SM (Section 4) — threads in different blocks cannot synchronize like this. Tiling trades a little code complexity for far fewer slow global-memory reads.


```python
%%writefile gemm_tiled.cu
#include <stdio.h>
#include <math.h>

#define TILE 16   // block is TILE x TILE threads; must divide N here

__global__ void fill(float *m, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] = val;
}

// Fill with a varied, deterministic pattern (for a stronger correctness check).
__global__ void fill_varied(float *m, int n, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] = (float)(((i + seed) * 1103515245u >> 16) % 7) - 3.0f;
}

// Naive version, kept here for a direct comparison.
__global__ void gemm_naive(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) acc += A[row*N+k] * B[k*N+col];
        C[row*N+col] = acc;
    }
}

// Tiled version: cooperatively stage TILE x TILE sub-tiles in shared memory.
__global__ void gemm_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];   // on-chip, shared by the whole block
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // Walk across the tiles along the k dimension.
    for (int t = 0; t < N / TILE; t++) {
        // Each thread loads one element of A's tile and one of B's tile.
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();               // wait until the whole tile is loaded

        for (int k = 0; k < TILE; k++) // compute from fast shared memory
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();               // wait before overwriting the tile
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

float time_kernel(void (*which)(const float*,const float*,float*,int),
                  const float*A,const float*B,float*C,int N,dim3 grid,dim3 block){
    which<<<grid,block>>>(A,B,C,N); cudaDeviceSynchronize();   // warmup
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    which<<<grid,block>>>(A,B,C,N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main() {
    int N = 10240;   // divisible by TILE (10240 / 16 = 640)
    size_t bytes = (size_t)N*N*sizeof(float);
    float *A,*B,*C,*Cref;
    cudaMallocManaged(&A,bytes); cudaMallocManaged(&B,bytes);
    cudaMallocManaged(&C,bytes); cudaMallocManaged(&Cref,bytes);
    int total=N*N, bs=256, nb=(total+bs-1)/bs;
    fill<<<nb,bs>>>(A,total,1.0f); fill<<<nb,bs>>>(B,total,1.0f); cudaDeviceSynchronize();

    dim3 block(TILE,TILE), grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);
    double work = 2.0*(double)N*N*N/1e9;

    float msN = time_kernel(gemm_naive, A,B,C,N, grid, block);
    float errN=0; for(int i=0;i<total;i++) errN=fmaxf(errN,fabsf(C[i]-(float)N));

    for(int i=0;i<total;i++) C[i]=0.0f;
    float msT = time_kernel(gemm_tiled, A,B,C,N, grid, block);
    float errT=0; for(int i=0;i<total;i++) errT=fmaxf(errT,fabsf(C[i]-(float)N));

    printf("GEMM N=%d, block %dx%d\n", N, TILE, TILE);
    printf("  naive : %.3f ms   (%6.0f GFLOP/s)   maxErr=%.1f\n", msN, work/(msN/1000.0), errN);
    printf("  tiled : %.3f ms   (%6.0f GFLOP/s)   maxErr=%.1f\n", msT, work/(msT/1000.0), errT);
    printf("  tiled speedup vs naive: %.2fx\n", msN/msT);

    // STRONGER CHECK: all-ones can't catch an indexing bug (any N-term sum is N).
    // Re-fill with VARIED values and confirm tiled matches naive element-for-element.
    fill_varied<<<nb,bs>>>(A,total,1); fill_varied<<<nb,bs>>>(B,total,7); cudaDeviceSynchronize();
    gemm_naive<<<grid,block>>>(A,B,Cref,N);
    gemm_tiled<<<grid,block>>>(A,B,C,N);
    cudaDeviceSynchronize();
    double maxDiff=0; for(int i=0;i<total;i++) maxDiff=fmax(maxDiff,fabs((double)C[i]-Cref[i]));
    printf("  varied-data check (tiled vs naive): max diff = %.3f  %s\n",
           maxDiff, maxDiff==0.0 ? "(identical -> indexing correct)" : "(MISMATCH!)");

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Cref);
    return 0;
}
```

    Overwriting gemm_tiled.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 gemm_tiled.cu -o gemm_tiled && ./gemm_tiled
```

    GEMM N=10240, block 16x16
      naive : 1935.822 ms   (  1109 GFLOP/s)   maxErr=0.0
      tiled : 1468.403 ms   (  1462 GFLOP/s)   maxErr=0.0
      tiled speedup vs naive: 1.32x
      varied-data check (tiled vs naive): max diff = 0.000  (identical -> indexing correct)


### How does this compare to a CPU?

Finally, the headline comparison: the same matrix multiply on the GPU versus a single CPU thread running the ordinary triple-nested loop. Both compute the identical result (and both are checked against the all-ones gold sample). The GPU time is measured kernel-only with CUDA events; the CPU time is wall-clock around its loop.

**Note on size:** the other cells in this section use N=10240, but this comparison drops to **N=1024**. That is deliberate — the single-threaded CPU loop scales as N^3, so at N=10240 it would run for *hours*. Even at the smaller N=1024 (1000x less work), the CPU still takes several seconds while the GPU finishes in about a millisecond. That gap *is* the lesson.


```python
%%writefile gemm_cpu_vs_gpu.cu
#include <stdio.h>
#include <math.h>
#include <time.h>

__global__ void fill(float *m, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] = val;
}
__global__ void gemm(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) acc += A[row*N+k] * B[k*N+col];
        C[row*N+col] = acc;
    }
}
void gemm_cpu(const float *A, const float *B, float *C, int N) {
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) acc += A[row*N+k] * B[k*N+col];
            C[row*N+col] = acc;
        }
}

int main() {
    int N = 1024;
    size_t bytes = (size_t)N*N*sizeof(float);
    double work = 2.0*(double)N*N*N/1e9;
    float *A,*B,*C;
    cudaMallocManaged(&A,bytes); cudaMallocManaged(&B,bytes); cudaMallocManaged(&C,bytes);
    int total=N*N, bs=256, nb=(total+bs-1)/bs;
    fill<<<nb,bs>>>(A,total,1.0f); fill<<<nb,bs>>>(B,total,1.0f); cudaDeviceSynchronize();
    printf("Matrix multiply, N=%d  (%.1f GFLOP of work)\n\n", N, work);

    // ---- GPU ----
    dim3 block(16,16), grid((N+15)/16,(N+15)/16);
    gemm<<<grid,block>>>(A,B,C,N); cudaDeviceSynchronize();   // warmup
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); gemm<<<grid,block>>>(A,B,C,N); cudaEventRecord(e); cudaEventSynchronize(e);
    float gms=0; cudaEventElapsedTime(&gms,s,e);
    float gerr=0; for(int i=0;i<total;i++) gerr=fmaxf(gerr,fabsf(C[i]-(float)N));
    printf("GPU  : %8.3f ms   (%7.0f GFLOP/s)   maxErr=%.1f\n", gms, work/(gms/1000.0), gerr);

    // ---- CPU (single thread, naive triple loop) ----
    float *Ccpu=(float*)malloc(bytes);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    gemm_cpu(A,B,Ccpu,N);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double cms=(t1.tv_sec-t0.tv_sec)*1000.0+(t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("CPU  : %8.1f ms   (%7.1f GFLOP/s)   (single thread, naive)\n", cms, work/(cms/1000.0));

    printf("\nGPU is about %.0fx faster than this CPU code --\n", cms/gms);
    printf("BUT this CPU baseline is deliberately weak: single-threaded, no vectorization,\n");
    printf("and a cache-hostile access pattern. A multi-threaded, cache-blocked CPU version\n");
    printf("(or a tuned BLAS library) would be far faster -- the honest GPU advantage on this\n");
    printf("kind of problem is more like tens of times, not thousands.\n");

    free(Ccpu); cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
```

    Overwriting gemm_cpu_vs_gpu.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 -O2 gemm_cpu_vs_gpu.cu -o gemm_cpu_vs_gpu && ./gemm_cpu_vs_gpu
```

    Matrix multiply, N=1024  (2.1 GFLOP of work)
    
    GPU  :    1.168 ms   (   1838 GFLOP/s)   maxErr=0.0
    CPU  :   7703.6 ms   (    0.3 GFLOP/s)   (single thread, naive)
    
    GPU is about 6593x faster than this CPU code --
    BUT this CPU baseline is deliberately weak: single-threaded, no vectorization,
    and a cache-hostile access pattern. A multi-threaded, cache-blocked CPU version
    (or a tuned BLAS library) would be far faster -- the honest GPU advantage on this
    kind of problem is more like tens of times, not thousands.


### Takeaways

- **High arithmetic intensity is what lets a GPU shine.** Matrix multiply does O(N^3) math on O(N^2) data, so values *can* be reused many times — unlike array addition, which used each value once and just streamed memory.
- **Block size matters more here.** Because the kernel has real work per byte, occupancy (driven by block size) affects how well the GPU hides latency — so the block-size sweep showed a clear difference, unlike with addition.
- **Shared-memory tiling reduces slow global-memory traffic** by loading each tile once and reusing it from fast on-chip memory. This is the single most important optimization pattern in CUDA.

**But notice how far from "compute-bound" we actually are.** The naive kernel reached only ~1100 GFLOP/s and the tiled kernel ~1500 — both just **3–5% of the L4's ~30 TFLOP/s FP32 peak**. Two tells that these kernels are still limited by memory/cache, not arithmetic: (1) that low percentage, and (2) the same naive kernel runs *faster* at N=1024 (~1800 GFLOP/s) than at N=10240 (~1100), purely because the smaller matrices fit in the L4's ~50 MB L2 cache and the larger ones don't. A truly compute-bound GEMM needs register blocking and Tensor Cores — which is exactly what NVIDIA's **cuBLAS** library does, running many times faster than either kernel here. Tiling is the first real step in that direction, not the finish line.

**Read the CPU speedup honestly.** The "≈6000×" headline compares against simple, single-threaded, cache-unfriendly CPU code, so it flatters the GPU enormously. A fair CPU baseline — multi-threaded, vectorized, cache-blocked, or a tuned BLAS — would close most of that gap, leaving the GPU's real advantage at roughly tens of times for this kind of problem. The goal of this section is to show *why* the GPU wins on high-arithmetic-intensity work and *how* the core optimization works — not to crown our teaching kernel a champion.

---
## 7. Error Handling

CUDA errors are **silent by default**. Your program may appear to work while producing garbage results. This is one of the most common sources of bugs in CUDA programs.

### Why CUDA Errors Are Silent

CUDA uses asynchronous execution - the CPU doesn't wait for GPU operations to complete. When you call a CUDA function:

1. The CPU queues the operation and continues immediately
2. The GPU executes it later
3. If it fails, the CPU has already moved on

This means errors can occur "in the background" without crashing your program.

### The cudaError_t Type

Every CUDA runtime API function returns a `cudaError_t` value. Let's see it in action:


```python
%%writefile check_error_type.cu
#include <stdio.h>

int main() {
    float *d_ptr;
    
    // cudaMalloc returns cudaError_t - let's capture and inspect it
    cudaError_t err = cudaMalloc(&d_ptr, 1024 * sizeof(float));
    
    printf("Return value: %d (cudaSuccess = 0)\n", err);
    printf("Error name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
    
    // Now let's trigger an error - try to allocate way too much memory
    printf("\n--- Triggering an error ---\n");
    cudaError_t bad_err = cudaMalloc(&d_ptr, (size_t)1024 * 1024 * 1024 * 1024);  // 1 TB!
    
    printf("Return value: %d\n", bad_err);
    printf("Error name: %s\n", cudaGetErrorName(bad_err));
    printf("Error description: %s\n", cudaGetErrorString(bad_err));
    
    cudaFree(d_ptr);
    return 0;
}
```

    Overwriting check_error_type.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 check_error_type.cu -o check_error_type && ./check_error_type
```

    Return value: 0 (cudaSuccess = 0)
    Error name: cudaSuccess
    Error description: no error
    
    --- Triggering an error ---
    Return value: 2
    Error name: cudaErrorMemoryAllocation
    Error description: out of memory


Key functions for error handling:
- `cudaGetErrorName(err)` - returns the error enum name (e.g., "cudaErrorMemoryAllocation")
- `cudaGetErrorString(err)` - returns a human-readable description

### Two Types of Errors

| Error Type | When Detected | How to Check |
|------------|---------------|-------------|
| **Synchronous** | Immediately (invalid arguments, allocation failures) | Check return value of the API call |
| **Asynchronous** | Later (kernel crashes, illegal memory access) | Call `cudaGetLastError()` or `cudaDeviceSynchronize()` |

**Kernel launches** (`kernel<<<...>>>()`) don't return `cudaError_t` directly - they queue work and return immediately. Use `cudaGetLastError()` to check if the launch itself failed, and `cudaDeviceSynchronize()` to catch errors during execution.

### Error Checking Macro

Writing error checks for every call is tedious. Use a macro:

**Note on `__FILE__` and `__LINE__`:** These are special variables built into the C/C++ compiler (called preprocessor macros). Before your code compiles, the compiler automatically replaces `__FILE__` with the current filename as a string, and `__LINE__` with the current line number. This happens at compile time, not runtime - so each error message shows exactly where in your code the problem occurred.

| Macro | Becomes | Type |
|-------|---------|------|
| `__FILE__` | the source filename, e.g. `"error_handling.cu"` | string |
| `__LINE__` | the line number where it appears | integer |

**Why a macro and not a function?** A macro is expanded *inline at each call site* before compilation. So when `cudaMalloc` on a given line fails, `__LINE__` reports *that* line — pinpointing exactly which call failed. If `CUDA_CHECK` were an ordinary function, `__LINE__` would always print the single line inside the function where the `fprintf` lives, which would be useless. In the code below, `__FILE__` and `__LINE__` appear once inside the macro definition, but they take on the right values at every `CUDA_CHECK(...)` call.


```python
%%writefile error_handling.cu
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
```

    Overwriting error_handling.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 error_handling.cu -o error_handling && ./error_handling
```

    Kernel executed successfully!


### When to Check for Errors

**Always check:**
- `cudaMalloc` / `cudaMallocManaged` - memory allocation can fail
- `cudaMemcpy` - data transfer errors
- After kernel launches - use `cudaGetLastError()` + `cudaDeviceSynchronize()`

**In production code:** Check every CUDA call. The overhead is negligible compared to GPU operations.

**During development:** At minimum, add `cudaDeviceSynchronize()` + error check after kernels to catch bugs early.

### Common Errors and Their Causes

| Error | Typical Cause |
|-------|---------------|
| `cudaErrorInvalidConfiguration` | Too many threads per block (max 1024) or invalid grid dimensions |
| `cudaErrorMemoryAllocation` | Requested more memory than available VRAM |
| `cudaErrorIllegalAddress` | Kernel accessed memory outside allocated region |
| `cudaErrorInvalidDevice` | Trying to use a GPU that doesn't exist |
| `cudaErrorNoKernelImageForDevice` | Compiled for wrong architecture (e.g., sm_80 code on sm_75 GPU) |
| `cudaErrorLaunchTimeout` | Kernel took too long (Windows display driver timeout) |

---
## 8. Memory Management

Memory is typically the bottleneck in GPU programs. Understanding memory types is essential.

### Unified Memory vs Explicit Memory

So far we've used **Unified Memory** (`cudaMallocManaged`) for simplicity. For production code, **explicit memory management** gives better performance.

| Approach | Pros | Cons |
|----------|------|------|
| Unified Memory | Simple, automatic | Hidden overhead, less control |
| Explicit | Maximum performance | More code, manual management |

### Explicit Memory Management (CPU Initialization)

This approach initializes data on the CPU, then copies it to the GPU. Use this when data comes from external sources (files, network, user input):


```python
%%writefile explicit_memory.cu
#include <stdio.h>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1 << 30;  // ~1 billion elements
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
```

    Overwriting explicit_memory.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 explicit_memory.cu -o explicit_memory && ./explicit_memory
```

    Max error: 0.000000


### GPU-Side Initialization (Better for Generated Data)

When data is generated algorithmically (constants, sequences, random numbers), initialize directly on the GPU. This avoids the CPU→GPU transfer entirely:

| Approach | Best For |
|----------|----------|
| CPU init + copy | Data from files, network, databases, user input |
| GPU init | Constants, patterns, random numbers, data already on GPU |


```python
%%writefile gpu_init.cu
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
```

    Overwriting gpu_init.cu



```bash
%%bash
/usr/local/cuda/bin/nvcc -arch=sm_89 gpu_init.cu -o gpu_init && ./gpu_init
```

    Max error: 0.000000


**Key difference:** The GPU initialization version skips the `cudaMemcpy` for input data entirely. For large arrays, this can significantly improve performance since CPU↔GPU transfers are often the bottleneck. In my case, it reduced the total time to 11 seconds. 

### GPU Memory Hierarchy

GPUs have several memory types with different speeds and scopes:

| Memory | Speed | Scope | Size (NVIDIA L4, sm_89) | Use Case |
|--------|-------|-------|------|----------|
| Registers | Fastest | Per thread | 64 K 32-bit registers per SM, split across its threads | Local variables |
| Shared Memory | Very fast | Per block | 48 KB per block by default; up to 99 KB with opt-in (`cudaFuncSetAttribute`). The SM has 100 KB total to share among its resident blocks. | Thread cooperation |
| L1 Cache | Fast | Per SM | shares a 128 KB pool per SM with shared memory | Hardware-managed |
| L2 Cache | Medium | Device-wide | ~50 MB | Hardware-managed |
| Global Memory (VRAM) | Slow | All threads | 23 GB | Main data storage |

> **Watch the per-block limit.** That 100 KB figure is the budget for a *whole SM*, not one block. A single block gets **48 KB by default**; asking for more (e.g. `__shared__ float buf[25000]`, ~98 KB) fails to launch with `cudaErrorLaunchOutOfResources` unless you explicitly opt in with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` and use *dynamic* shared memory (max 99 KB on sm_89).

For beginners, focus on global memory. Shared memory optimization is an intermediate topic.

---
## 9. Profiling Your Code

**Nsight Systems** (`nsys`) profiles CPU/GPU activity and shows where time is spent.

### Basic Profiling


```bash
%%bash
nsys profile --stats=true ./add_gpu_v3 2>&1 | grep -A 10 'cuda_gpu_kern_sum'
```

    [6/8] Executing 'cuda_gpu_kern_sum' stats report
    
     Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)             Name           
     --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  --------------------------
        100.0          1558137          1  1558137.0  1558137.0   1558137   1558137          0.0  add(int, float *, float *)
    
    [7/8] Executing 'cuda_gpu_mem_time_sum' stats report
    
     Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)               Operation              
     --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------------
         64.9         20116269    128  157158.4  157151.0    157087    157471         68.7  [CUDA memcpy Unified Host-to-Device]


### Understanding the Output

| Column | Meaning |
|--------|--------|
| Time (%) | Percentage of total GPU time |
| Total Time (ns) | Kernel execution time in nanoseconds |
| Instances | Number of kernel launches |
| Name | Kernel function name |

To convert nanoseconds to seconds: divide by 1,000,000,000 (10^9).

### Comparing Versions

Let's profile all three versions to see the speedup:


```bash
%%bash
echo "=== Version 1: 1 thread ==="
nsys profile --stats=true ./add_gpu_v1 2>&1 | grep -A 5 'cuda_gpu_kern_sum'

echo ""
echo "=== Version 2: 256 threads (1 block) ==="
nsys profile --stats=true ./add_gpu_v2 2>&1 | grep -A 5 'cuda_gpu_kern_sum'

echo ""
echo "=== Version 3: Many blocks x 256 threads ==="
nsys profile --stats=true ./add_gpu_v3 2>&1 | grep -A 5 'cuda_gpu_kern_sum'
```

    === Version 1: 1 thread ===
    [6/8] Executing 'cuda_gpu_kern_sum' stats report
    
     Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)             Name           
     --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  --------------------------
        100.0      89023173952          1  89023173952.0  89023173952.0  89023173952  89023173952          0.0  add(int, float *, float *)
    
    
    === Version 2: 256 threads (1 block) ===
    [6/8] Executing 'cuda_gpu_kern_sum' stats report
    
     Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)             Name           
     --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  --------------------------
        100.0       3280318336          1  3280318336.0  3280318336.0  3280318336  3280318336          0.0  add(int, float *, float *)
    
    
    === Version 3: Many blocks x 256 threads ===
    [6/8] Executing 'cuda_gpu_kern_sum' stats report
    
     Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)             Name           
     --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  --------------------------
        100.0          1539004          1  1539004.0  1539004.0   1539004   1539004          0.0  add(int, float *, float *)
    


---
## 10. Common Pitfalls

### 1. Forgetting to Synchronize

```c
// WRONG: Results may not be ready
add<<<blocks, threads>>>(N, x, y);
printf("%f\n", y[0]);  // Race condition!

// CORRECT
add<<<blocks, threads>>>(N, x, y);
cudaDeviceSynchronize();  // Wait for GPU
printf("%f\n", y[0]);    // Safe
```

### 2. Out-of-Bounds Access

When total threads exceed array size, add bounds checking:

```c
__global__ void add(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)  // Bounds check!
        y[i] = x[i] + y[i];
}
```

### 3. Integer Overflow in Index Calculation

For very large arrays, use `size_t` or `long long`:

```c
__global__ void process(size_t n, float *data) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // ...
}
```

### 4. Not Checking Errors

Always use error checking (see Section 7). Silent failures are common.

### 5. Wrong Architecture Flag

```bash
# If your GPU is compute capability 7.5 (T4, RTX 2080)
nvcc -arch=sm_75 program.cu -o program  # CORRECT
nvcc -arch=sm_80 program.cu -o program  # Compiles but may fail at runtime
```

---
## 11. Summary

### CPU to CUDA Cheat Sheet

| Concept | CPU (C) | GPU (CUDA) |
|---------|---------|------------|
| Function declaration | `void func()` | `__global__ void func()` |
| Memory allocation | `malloc(size)` | `cudaMallocManaged(&ptr, size)` |
| Memory free | `free(ptr)` | `cudaFree(ptr)` |
| Function call | `func(args)` | `func<<<blocks, threads>>>(args)` |
| Wait for completion | (automatic) | `cudaDeviceSynchronize()` |
| Thread ID | N/A | `blockIdx.x * blockDim.x + threadIdx.x` |
| File extension | `.c` | `.cu` |
| Compiler | `gcc` | `nvcc` |

### Key Concepts

1. **GPUs excel at data parallelism** - same operation on many elements
2. **Threads are organized hierarchically** - threads → blocks → grid
3. **Each thread computes its global index** - `blockIdx.x * blockDim.x + threadIdx.x`
4. **Use grid-stride loops** for flexible, efficient kernels
5. **Always check for errors** - CUDA fails silently
6. **Profile before optimizing** - measure, don't guess

---
## Resources

**Documentation:**
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)

**Free Courses:**
- [Fundamentals of Accelerated Computing with CUDA C/C++](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/about) - NVIDIA DLI
- [Fundamentals of Accelerated Computing with CUDA Python](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-02+V1/about) - NVIDIA DLI

**Tools:**
- `nsys` - Nsight Systems profiler (used in this guide)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) - Visual profiler
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) - Kernel profiler

---
<a id="appendix-a-setup"></a>
## Appendix A: Setup

This appendix covers installing the CUDA development environment. Skip if already set up.

### Requirements

- NVIDIA GPU (any CUDA-capable GPU)
- Linux (Ubuntu 22.04/24.04 recommended)
- C++ compiler (g++)
- Python + Jupyter (for this notebook)

### A.1 Install Python Environment

You need Python to run Jupyter notebooks. Several options exist:

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Miniconda** | Lightweight, conda package manager, easy env management | Separate from system Python | Data science, ML projects |
| **Anaconda** | Pre-installed packages, GUI tools | Large download (~3GB) | Beginners who want everything included |
| **System Python + pip** | Already installed, simple | Can conflict with system packages | Quick scripts, minimal setup |
| **pyenv + pip** | Multiple Python versions, clean isolation | More setup steps | Developers managing multiple projects |

We use **Miniconda** here because:
- Conda handles complex dependencies (like CUDA libraries) better than pip
- Easy to create isolated environments for different projects
- Lightweight compared to full Anaconda

**Skip this if you already have Python + Jupyter working.**


```bash
%%bash
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    $HOME/miniconda3/bin/conda init bash
    echo "Miniconda installed. Run: source ~/.bashrc"
else
    echo "Miniconda already installed"
fi
```

    Miniconda already installed


### A.2 Install Jupyter Kernel

A Jupyter **kernel** is the backend that executes code in notebook cells. Each kernel connects a specific Python environment to Jupyter. Without `ipykernel` installed in your conda environment, Jupyter won't be able to run Python code from that environment.

**Skip this if:** You can already run Python cells in Jupyter notebooks.


```bash
%%bash
# A %%bash cell runs a NON-interactive shell, which does not source ~/.bashrc,
# so `conda` is usually not on PATH here. Call it by absolute path instead
# (same reason we call nvcc as /usr/local/cuda/bin/nvcc throughout this notebook).
CONDA="$HOME/miniconda3/bin/conda"
if ! ("$CONDA" list -n base ipykernel 2>/dev/null | grep -q ipykernel); then
    "$CONDA" install -n base ipykernel --update-deps --force-reinstall -y
else
    echo "ipykernel already installed"
fi
```

    ipykernel already installed


### A.3 Install CUDA Toolkit (Ubuntu)

The CUDA Toolkit provides:
- `nvcc` compiler
- CUDA runtime libraries
- Header files
- Profiling tools (Nsight Systems, Nsight Compute)


```bash
%%bash
# For Ubuntu 24.04 with CUDA 13.1 (current version)
if ! command -v /usr/local/cuda/bin/nvcc &> /dev/null; then
    wget -nc https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-13-1
else
    echo "CUDA toolkit already installed"
fi
```

    CUDA toolkit already installed


### A.4 Add CUDA to PATH


```bash
%%bash
if ! grep -q 'cuda' ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "Added CUDA to PATH. Run: source ~/.bashrc"
else
    echo "CUDA PATH already configured"
fi
```

    CUDA PATH already configured


### A.5 Install NVIDIA Driver

Choose ONE option based on your GPU:

**Option 1: Open-source driver** (recommended for datacenter GPUs like T4, V100, A100)


```bash
%%bash
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt-get install -y nvidia-open
else
    echo "NVIDIA driver already installed"
fi
```

    NVIDIA driver already installed


**Option 2: Proprietary driver** (for consumer GPUs like RTX series)


```bash
%%bash
# Uncomment to use proprietary driver instead
# sudo apt-get install -y cuda-drivers
```

### A.6 Verify Installation


```bash
%%bash
echo "Checking installation:"
command -v g++ >/dev/null && echo "  g++: installed" || echo "  g++: NOT FOUND"
/usr/local/cuda/bin/nvcc --version >/dev/null 2>&1 && echo "  nvcc: installed" || echo "  nvcc: NOT FOUND"
command -v nvidia-smi >/dev/null && echo "  nvidia-smi: installed" || echo "  nvidia-smi: NOT FOUND"
echo ""
nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null || echo "GPU not accessible"
```

    Checking installation:
      g++: installed
      nvcc: installed
      nvidia-smi: installed
    
    name, driver_version
    NVIDIA L4, 610.43.02


## Good luck!
