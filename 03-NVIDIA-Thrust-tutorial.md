# NVIDIA Thrust Library: A Comprehensive Tutorial

> **Note**: This tutorial is located in the `00-Pre-requisite` directory as it covers foundational concepts needed before diving into the main CUDA learning materials.

## Introduction

**Thrust** is a C++ template library for CUDA based on the Standard Template Library (STL). It lets you express many GPU computations as operations on ranges, instead of writing CUDA kernels by hand. If you are comfortable with C++ containers, iterators, and callable objects, Thrust should feel like a GPU-oriented version of the STL.

A useful mental model is:

1. Put data in a container: `thrust::host_vector` for CPU memory, `thrust::device_vector` for GPU memory.
2. Describe a range using iterators: `begin()` and `end()` mark the data to process.
3. Choose an algorithm: `transform`, `reduce`, `sort`, `scan`, and similar primitives.
4. Provide an operation when needed: a built-in function object like `thrust::plus<T>()`, or your own functor.
5. Keep data on the GPU until you actually need results on the CPU.

Thrust does not remove the need to understand GPU programming. Host/device memory, data transfer cost, parallelism, and synchronization still matter. What Thrust gives you is a higher-level way to use those concepts without writing a custom kernel for every operation.

### Why Use Thrust?

- **High-level abstractions**: Write GPU code without dealing with low-level CUDA kernel details
- **STL-like interface**: Familiar API for C++ developers
- **Productivity**: Implement complex parallel algorithms in fewer lines of code
- **Performance**: Highly optimized implementations that rival hand-written CUDA kernels
- **Portability**: Code can run on both CPU and GPU backends, depending on how it is configured

### Key Features

1. **Containers**: `host_vector` and `device_vector` for managing data
2. **Algorithms**: Transform, reduce, scan, sort, and more
3. **Iterators**: Fancy iterators for advanced data access patterns
4. **Functors**: Custom operations that work seamlessly with algorithms

### How This Tutorial Builds Up

The examples are ordered so each section adds one idea:

1. Vectors teach where data lives: CPU memory vs GPU memory.
2. Basic algorithms teach how Thrust launches parallel work for common operations.
3. Functors teach how to pass custom behavior into those algorithms.
4. Fancy iterators teach how to avoid unnecessary temporary arrays.
5. The final examples combine these pieces into useful numerical patterns and performance measurements.

By the end, you should be able to look at a Thrust expression and answer three questions: what data range is being processed, what operation is applied to each element or range, and where the expensive host/device transfers happen.

---

## Setup

This notebook assumes you have:
- CUDA toolkit installed
- A CUDA-capable NVIDIA GPU
- Compiler that supports C++17 or later (Thrust 3.x requires C++17)
- Working knowledge of C++ templates, iterators, and simple callable objects

For compiling Thrust code from this notebook, we'll use `nvcc` with appropriate flags.



```python
# Check CUDA availability
!nvcc --version
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2025 NVIDIA Corporation
    Built on Tue_Dec_16_07:23:41_PM_PST_2025
    Cuda compilation tools, release 13.1, V13.1.115
    Build cuda_13.1.r13.1/compiler.37061995_0


---

## 1. Vectors: The Foundation

Thrust provides two primary container types:

- **`thrust::host_vector<T>`**: Resides in CPU (host) memory
- **`thrust::device_vector<T>`**: Resides in GPU (device) memory

Both containers behave like `std::vector` from the C++ STL:
- Generic (can store any data type)
- Dynamically resizable
- Support element access via `[]` operator
- Automatic memory management

### Example 1.1: Basic Vector Operations


```python
%%writefile thrust_vectors_basic.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

int main() {
    // Create a host_vector with 5 elements
    thrust::host_vector<int> H(5);
    
    // Initialize elements
    H[0] = 10;
    H[1] = 20;
    H[2] = 30;
    H[3] = 40;
    H[4] = 50;
    
    std::cout << "Host vector contents: ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    // Copy host_vector to device_vector
    thrust::device_vector<int> D = H;
    
    // Resize device vector
    D.resize(10);
    D[5] = 60;
    D[6] = 70;
    
    std::cout << "Device vector size: " << D.size() << std::endl;
    
    // Copy back to host to verify
    H = D;
    
    std::cout << "Updated host vector: ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

    Overwriting thrust_vectors_basic.cu



```python
# Compile and run
!nvcc -arch=sm_89 thrust_vectors_basic.cu -o thrust_vectors_basic
!./thrust_vectors_basic
```

    Host vector contents: 10 20 30 40 50 
    Device vector size: 10
    Updated host vector: 10 20 30 40 50 60 70 0 0 0 


### Example 1.2: Vector Initialization Methods

---

### Important: Why Do We Copy Vectors Back to Host?

Look at the following code:

```cpp
thrust::device_vector<int> D(10);        // Data on GPU
thrust::sequence(D.begin(), D.end());    // GPU operation

thrust::host_vector<int> H = D;          // Copy GPU → CPU
std::cout << H[i];                       // Print from CPU
```

**Why can't we just print directly from `device_vector`?**

#### GPU and CPU Have Separate Memory

```
┌─────────────────┐          ┌─────────────────┐
│   CPU (Host)    │          │   GPU (Device)  │
│                 │          │                 │
│  Host Memory    │          │  Device Memory  │
│  (RAM)          │          │  (VRAM)         │
│                 │          │                 │
│  ✅ CPU can     │          │  ✅ GPU can     │
│     access      │          │     access      │
│                 │          │                 │
│  ❌ GPU cannot  │          │  ❌ CPU cannot  │
│     access      │          │     access      │
└─────────────────┘          └─────────────────┘
         │                            │
         └────────────────────────────┘
              Connected by PCIe bus
             (copying data is SLOW)
```

**The Problem:**
- `device_vector` stores data in GPU memory (VRAM)
- `std::cout` runs on the CPU
- CPU code **cannot directly access GPU memory**

**The Solution:**
```cpp
thrust::host_vector<int> H = D;  // Copy data from GPU to CPU
```

This copies the data across the PCIe bus from GPU memory to CPU memory.

---

#### Performance Implication

**Copying between host and device is EXPENSIVE!**

```cpp
// ❌ BAD: Copy back and forth repeatedly
for (int i = 0; i < 1000; i++) {
    thrust::device_vector<float> D(1000000);
    // ... do GPU work ...
    thrust::host_vector<float> H = D;  // SLOW!
    std::cout << H[0] << std::endl;
}
```

```cpp
// ✅ GOOD: Keep data on GPU, only copy when necessary
thrust::device_vector<float> D(1000000);
for (int i = 0; i < 1000; i++) {
    // ... do ALL GPU work ...
}
thrust::host_vector<float> H = D;  // Copy ONCE at the end
std::cout << H[0] << std::endl;
```

**Best Practice:**
1. **Move data to GPU once**
2. **Do all computations on GPU**
3. **Copy results back to CPU only when you need them** (printing, saving to file, etc.)

This is why GPU programming is called "data-parallel computing" - you want to process lots of data in parallel on the GPU and minimize data transfers!

---


```python
%%writefile thrust_vector_init.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

int main() {
    // Different initialization methods
    
    // 1. Default initialization (size 10, all zeros)
    thrust::device_vector<int> D1(10);
    
    // 2. Initialize with a specific value
    thrust::device_vector<int> D2(10, 5);  // 10 elements, all set to 5
    
    // 3. Initialize from another vector
    thrust::device_vector<int> D3(D2);
    
    // 4. Initialize from host vector
    thrust::host_vector<int> H(5, 100);
    thrust::device_vector<int> D4 = H;
    
    // 5. Initialize using iterators
    thrust::host_vector<int> H2(5);
    H2[0] = 1; H2[1] = 2; H2[2] = 3; H2[3] = 4; H2[4] = 5;
    thrust::device_vector<int> D5(H2.begin(), H2.end());
    
    // Copy back and display
    thrust::host_vector<int> result = D5;
    std::cout << "D5 contents: ";
    for(int i = 0; i < result.size(); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

    Overwriting thrust_vector_init.cu



```python
!nvcc -arch=sm_89 thrust_vector_init.cu -o thrust_vector_init
!./thrust_vector_init
```

    D5 contents: 1 2 3 4 5 


---

## 2. Basic Algorithms

Thrust provides a rich set of parallel algorithms that operate on vectors. These algorithms are similar to STL algorithms but execute in parallel on the GPU.

### 2.1 Fill and Sequence

- **`thrust::fill`**: Set all elements to a specific value
- **`thrust::sequence`**: Generate a sequence of values


```python
%%writefile thrust_fill_sequence.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <iostream>

int main() {
    thrust::device_vector<int> D(10);
    
    // Fill all elements with 7
    thrust::fill(D.begin(), D.end(), 7);
    
    thrust::host_vector<int> H = D;
    std::cout << "After fill(7): ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    // Generate sequence: 0, 1, 2, 3, ...
    thrust::sequence(D.begin(), D.end());
    
    H = D;
    std::cout << "After sequence(): ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    // Generate sequence with start value and step
    // sequence(start, end, init_value, step)
    thrust::sequence(D.begin(), D.end(), 10, 5);  // 10, 15, 20, 25, ...
    
    H = D;
    std::cout << "After sequence(10, 5): ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

    Overwriting thrust_fill_sequence.cu



```python
!nvcc -arch=sm_89 thrust_fill_sequence.cu -o thrust_fill_sequence
!./thrust_fill_sequence
```

    After fill(7): 7 7 7 7 7 7 7 7 7 7 
    After sequence(): 0 1 2 3 4 5 6 7 8 9 
    After sequence(10, 5): 10 15 20 25 30 35 40 45 50 55 


### 2.2 Transform

**`thrust::transform`** applies a unary or binary operation to elements.

Syntax:
```cpp
thrust::transform(input_begin, input_end, output_begin, operation);
```

### Functors: Passing Behavior into Algorithms

Most Thrust algorithms take an **operation** as their last argument. That operation is
usually a **functor** (function object): a small `struct` with an `operator()`, optionally
carrying state. If you know `std::function`, lambdas, or STL function objects, this is the
same idea, with one CUDA-specific requirement.

There is a **division of labor** worth fixing in your mind now, because it applies to *every*
Thrust algorithm that takes an operation (`transform`, `reduce`, `sort`, and the rest):

- The **algorithm** decides *which values* get passed in and *when* — it walks the ranges,
  fetches elements, and calls your operation, supplying the arguments each time.
- The **operation** (functor) decides *what to compute* from those values — it never chooses
  the data, only what to do with it.

So when you write `multiply_by(3.0f)` (the example just below) you are only configuring the
operation — storing the factor `3` — you are *not* handing it any data. The algorithm feeds
the per-element values in later. Keep this split — algorithm supplies the values, functor
defines the math — in view through the next two sections; it is the single idea behind both
`transform` and `reduce`.

**Stateless functor** — behaves like a plain function:

```cpp
struct square {                          // a functor type with no stored data
    __host__ __device__                  // operator() is callable from CPU and GPU code
    float operator()(float x) const {    // called once per element; x is that element
        return x * x;                    // the behavior: square the input
    }
};
```

**Stateful functor** — captures a value at construction and reuses it per element:

```cpp
struct multiply_by {
    float factor;                              // data member: the captured state (each object has its own)
    multiply_by(float f) : factor(f) {}        // constructor: copy f into member 'factor'; runs once, on the host
    __host__ __device__                        // make operator() usable on both CPU and GPU
    float operator()(float x) const {          // const member function: called once per element
        return x * factor;                     // x = the input element, factor = the stored state
    }
};

// Build a functor whose factor = 3, then apply it to every element of D: x -> x*3
thrust::transform(D.begin(), D.end(), out.begin(), multiply_by(3.0f));
```

A quick term: a **member** is anything declared inside the `struct`. Here `factor` is a
*data member* — a value each object owns (like `self.factor` on a Python object) — while the
constructor and `operator()` are *member functions*. Members are reached through an object
with the dot operator (`m.factor`, `m(10.0f)`), and every object gets its **own copy** of the
data members, so `multiply_by(3.0f)` and `multiply_by(7.0f)` hold independent `factor` values.

Read it in two stages, because the constructor and `operator()` run at very different times:

1. **Construction (once, on the host).** `multiply_by(3.0f)` builds one object and stores
   `3.0f` in its `factor` member. The part after the colon, `: factor(f)`, is a **member
   initializer list**: it initializes the member `factor` *to the value of* the constructor
   argument `f`. So `multiply_by(3.0f)` means `f = 3.0f`, which makes `factor = 3.0f`.
   Nothing is computed on the GPU yet — you have created a small object that *carries a 3
   around with it*.

   This is *initialization*, not *assignment*: the member is constructed directly from `f`
   **before** the constructor body `{}` runs, so `factor` is born holding `3.0f` rather than
   being default-created and then overwritten. Compare the two forms:

   ```cpp
   multiply_by(float f) : factor(f) {}     // initialize factor from f, in one step
   multiply_by(float f) { factor = f; }    // default-construct factor, THEN assign f to it
   ```

   For a `float` the result is identical and the choice is just style. But the initializer
   list becomes **mandatory** for `const` members and references (you cannot assign to them
   in the body) and for members that have no default constructor. The `saxpy_functor` later
   in this tutorial relies on exactly this: its `const float a;` can only be set as
   `: a(_a)`.

2. **Application (once per element, in parallel).** `transform` then calls that object's
   `operator()` for each input value. So with `D = [1, 2, 3, 4, 5]`:

   ```text
   operator()(1) -> 1 * factor -> 1 * 3 -> 3
   operator()(2) -> 2 * factor -> 2 * 3 -> 6
   operator()(3) -> 3 * factor -> 3 * 3 -> 9
   operator()(4) -> 4 * factor -> 4 * 3 -> 12
   operator()(5) -> 5 * factor -> 5 * 3 -> 15
   ```

   `factor` is read on every call but never changes — the input `x` varies, the captured
   state stays fixed. That is the whole point: the functor *remembers* `3` so you do not
   have to pass it in separately for each element.

**Why this matters on a GPU.** `transform` takes the functor **by value**, so the object you
build on the host is *copied* into the algorithm and ends up living in each GPU thread that
runs `operator()`. The captured `factor` therefore rides along to the device automatically —
this is how you get a runtime value (a scalar, a threshold, a learning rate) into thousands
of parallel threads without any explicit `cudaMemcpy`. Two practical consequences:

- Keep the captured state **small and copyable** (scalars, a few values). Do not store a
  host pointer or a `std::vector` and dereference it inside `operator()`; that address is
  meaningless on the device. To use a whole array on the device, pass a device pointer
  (see the `raw_pointer_cast` example in Section 5.5).
- The trailing `const` in `float operator()(float x) const` makes this a **const member
  function** — a promise that calling it does not modify the object. Inside such a method
  the members are read-only: you may read `factor`, but `factor = 5.0f;` would not compile.
  (This is a different `const` from the `const float a;` member in `saxpy_functor` below:
  trailing `const` after the parameter list constrains the *method*; `const` on a member
  declaration constrains the *member*. Note it also does not constrain the input `x`, which
  is passed by value.) It matters here because every thread shares its own copy of the
  functor and only *reads* the captured state. A functor whose `operator()` mutates a member
  is a red flag in parallel code: the threads would race on it, and the result would be
  undefined.

Changing the captured value just means constructing a different object — `multiply_by(10.0f)`
is an independent functor with `factor = 10`. State is what makes a functor more than a plain
function pointer: `saxpy_functor(a)` later in this tutorial carries the scalar `a` exactly the
same way.

**The one CUDA rule:** any functor that runs on the GPU must mark `operator()` with
`__host__ __device__`. Without it, the code is host-only and the device algorithm will not
compile. `__host__` = compiled for the CPU, `__device__` = compiled for the GPU; both
together = usable in either place.

> **What about lambdas?** A device lambda works too, but only when `nvcc` is invoked with
> the `--extended-lambda` flag and the lambda is annotated, e.g.
> `[] __device__ (float x) { return x*x; }`. Without that flag `nvcc` reports
> *"`__host__` or `__device__` annotation on lambda requires --extended-lambda"*. Named
> functors need no special flag, which is why this tutorial uses them.


```python
%%writefile thrust_transform.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <iostream>

// Custom functor to square a number
struct square {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};

int main() {
    thrust::device_vector<float> D(5);
    thrust::sequence(D.begin(), D.end(), 1.0f);  // 1, 2, 3, 4, 5
    
    // Transform using custom functor
    thrust::device_vector<float> D_squared(5);
    thrust::transform(D.begin(), D.end(), D_squared.begin(), square());
    
    thrust::host_vector<float> H = D_squared;
    std::cout << "Squared values: ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    // Transform using built-in functor (negate)
    thrust::device_vector<float> D_negated(5);
    thrust::transform(D.begin(), D.end(), D_negated.begin(), 
                     thrust::negate<float>());
    
    H = D_negated;
    std::cout << "Negated values: ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

    Overwriting thrust_transform.cu



```python
!nvcc -arch=sm_89 thrust_transform.cu -o thrust_transform
!./thrust_transform
```

    Squared values: 1 4 9 16 25 
    Negated values: -1 -2 -3 -4 -5 


### 2.3 Reduce

**`thrust::reduce`** collapses a whole range down to a single value by repeatedly combining
elements with a binary operation. With no operation specified it adds, so it computes the sum.

Common use cases:
- Sum of all elements
- Product of all elements
- Finding min/max

#### Reading the four-argument call

The product example uses the most general form:

```cpp
int product = thrust::reduce(D2.begin(), D2.end(), 1, thrust::multiplies<int>());
//                           \______range______/  ^   \___ binary operation ___/
//                                                 |
//                                            initial value (init)
```

The four arguments are: the **range** to reduce (`begin`, `end`), an **initial value**, and a
**binary operation** — a functor taking two values and returning one. `thrust::multiplies<int>()`
is a built-in functor (from `<thrust/functional.h>`) that returns `a * b`; the family also
includes `thrust::plus`, `thrust::minus`, `thrust::maximum`, and `thrust::minimum`. Reduce
starts from `init` and folds each element in with that operation:

```text
result = init
result = result * D2[0]   // 1 * 1 = 1
result = result * D2[1]   // 1 * 2 = 2
result = result * D2[2]   // 2 * 3 = 6
result = result * D2[3]   // 6 * 4 = 24
result = result * D2[4]   // 24 * 5 = 120
```

(In parallel, Thrust splits the range across threads, reduces each piece, then combines the
partial results with the same operation. The operation should be associative so the grouping
does not change the answer — `+`, `*`, `min`, `max` all qualify.)

#### Why `init` is `1` for a product

`init` is genuinely folded into the result, not just a starting placeholder. So it must be the
**identity** for the operation — the value that leaves the result unchanged:

- For a **sum**, the identity is `0` (`x + 0 == x`), which is why `init`-less `reduce` and
  `reduce(..., 0, thrust::plus<int>())` agree.
- For a **product**, the identity is `1` (`x * 1 == x`). Passing `1` gives `120`. Passing a
  different value scales the answer (`init = 2` would yield `240`), and passing `0` would
  give `0`, because anything times zero is zero — `0` is the identity for `+` but destroys a
  product. Choosing the right identity for the chosen operation is the key idea here.

#### `reduce` only does what the operation says

A common point of confusion: why does `thrust::maximum<int>()` return the largest element
instead of summing the elements like `multiplies` multiplied them? The answer is that
`reduce` has no built-in idea of "sum." It only ever does this:

```text
result = init
result = op(result, element)   // for each element, in turn
```

The **operation `op` decides what "combine" means** — `reduce` just applies it. Swapping the
operation is the only difference between these calls over `1..10`:

| Operation | `op(a, b)` returns | Result |
|-----------|--------------------|--------|
| `thrust::plus<int>()` (or none) | `a + b` | `55` (the sum) |
| `thrust::multiplies<int>()` | `a * b` | the product |
| `thrust::maximum<int>()` | the larger of `a` and `b` | `10` |

So `maximum` does not add anything — it keeps the bigger of the two values at each step and
discards the smaller one:

```text
result = 0                 // init
result = max(0, 1)  = 1
result = max(1, 2)  = 2
result = max(2, 3)  = 3
...
result = max(9, 10) = 10
```

Summing would require `op` to be `+`, which is exactly the default `reduce` — that is why
`reduce(D.begin(), D.end())` gives `55`. The element values are never accumulated unless the
operation you pass accumulates them.

**Caveat on `init = 0` for maximum.** The identity for `maximum` is "a value no element can
exceed," i.e. the smallest possible `int`. Here `0` works *only because every element is
≥ 0*: `max(0, x)` never changes the result for non-negative data. But if the vector could
contain negative numbers, `init = 0` would wrongly report `0` as the maximum. The safe
identity is `std::numeric_limits<int>::min()` (from `<limits>`) — just as `0` is the identity
for `+` and `1` for `*`.

#### How this squares with "`transform` passes values to the functor"

It can look contradictory to say *`reduce` does not do the addition* but also *`transform`
feeds `x` and `y` into the functor*. Both are true, and they describe the two halves of the
division of labor from the functor section — they are not in conflict:

- **The algorithm supplies the values.** `reduce` fetches each element and feeds it to the
  operation, exactly as `transform` fetches `X[i]` and `Y[i]` and feeds them in. Both
  algorithms deliver the data.
- **The operation defines the math.** Neither `reduce` nor `transform` has any built-in
  notion of "add" or "multiply." The functor you pass decides that — `thrust::plus` adds,
  `thrust::maximum` keeps the larger, `saxpy_functor` computes `a*x + y`.

```text
reduce    -> op(acc, element)  : algorithm picks acc & element; op decides it means "+"
transform -> op(x, y)          : algorithm picks X[i] & Y[i];  op decides it means "a*x+y"
```

The only real difference is the **shape** of the loop, not who does what:

- `transform` maps N inputs to N outputs — it calls the operation once per element and stores
  each result.
- `reduce` folds N inputs down to one value — it calls the operation repeatedly and feeds the
  running result back in as the next call's accumulator (`acc = op(acc, element)`). That
  feedback is why `reduce`'s operation takes `(accumulator, element)` and why the `init`
  value matters, while `transform`'s simply takes the input element(s).


```python
%%writefile thrust_reduce.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <iostream>

int main() {
    thrust::device_vector<int> D(10);
    thrust::sequence(D.begin(), D.end(), 1);  // 1, 2, 3, ..., 10
    
    // Sum all elements (default operation is addition)
    int sum = thrust::reduce(D.begin(), D.end());
    std::cout << "Sum of 1 to 10: " << sum << std::endl;
    
    // Sum with initial value
    int sum_with_init = thrust::reduce(D.begin(), D.end(), 100);  // 100 + sum
    std::cout << "Sum with init value 100: " << sum_with_init << std::endl;
    
    // Product of all elements
    thrust::device_vector<int> D2(5);
    thrust::sequence(D2.begin(), D2.end(), 1);  // 1, 2, 3, 4, 5
    
    int product = thrust::reduce(D2.begin(), D2.end(), 1, thrust::multiplies<int>());
    std::cout << "Product of 1 to 5: " << product << " (5! = 120)" << std::endl;
    
    // Maximum element
    int max_val = thrust::reduce(D.begin(), D.end(), 0, 
                                 thrust::maximum<int>());
    std::cout << "Maximum value: " << max_val << std::endl;
    
    return 0;
}
```

    Overwriting thrust_reduce.cu



```python
!nvcc -arch=sm_89 thrust_reduce.cu -o thrust_reduce
!./thrust_reduce
```

    Sum of 1 to 10: 55
    Sum with init value 100: 155
    Product of 1 to 5: 120 (5! = 120)
    Maximum value: 10


### 2.4 Execution Policies: Choosing CPU or GPU

The introduction said Thrust can run on either the CPU or the GPU. Execution policies are
how you say which. Passing `thrust::device` as the first argument runs the algorithm on the
GPU; `thrust::host` runs it on the CPU. (`thrust::par` and `thrust::seq` express parallel
vs. sequential intent for host backends such as OpenMP or TBB, when Thrust is configured to
use them.)

When you call an algorithm **without** a policy, Thrust picks one from the iterators: a
`device_vector` range implies device execution, a `host_vector` range implies host
execution. That is why every earlier example "just worked" on the GPU — the
`device_vector` chose the policy for us. The call below makes the choice explicit so the
two paths are visible side by side.

#### What "host backends such as OpenMP or TBB" means

A **backend** (Thrust also calls it a *system*) is the engine that actually executes an
algorithm. Thrust separates *what* you ask for (`reduce`, `sort`, …) from *where* it runs,
and there are two slots:

- **Device backend** — handles the work you think of as "on the GPU." In NVIDIA's Thrust the
  only GPU backend is **CUDA**, and it is the default; there is no menu of GPU vendors to pick
  from. The device slot *can* be redirected to a CPU engine (OMP, TBB, or serial CPP), but
  that runs the "device" work on the CPU — useful for debugging or building without a GPU, not
  for targeting different accelerator hardware.
- **Host backend** — runs on the CPU. This is what the phrase above refers to.

> **Other hardware (AMD, AWS Trainium, …).** These are *not* selectable Thrust backends.
> Thrust is an NVIDIA project (part of CCCL) and only speaks CUDA. AMD GPUs use a separate
> ported library, **rocThrust**, which mirrors the same API on top of AMD's HIP toolchain.
> AWS Trainium is a different model entirely — programmed through the **AWS Neuron SDK** via
> ML frameworks, with no Thrust path. For one source base spanning NVIDIA, AMD, and CPUs,
> people reach for vendor-neutral libraries such as Kokkos or standard C++ parallel
> algorithms instead.

When an algorithm executes on the host (because you passed `thrust::host`, or the data is a
`host_vector`), the host backend is the code that does the CPU-side work. There are three
choices:

| Host backend | What it is | CPU behavior |
|--------------|-----------|--------------|
| **CPP** (default) | Plain serial C++ | Single-threaded |
| **OMP** | **OpenMP**, a compiler-built-in standard for CPU threading | Uses many CPU cores |
| **TBB** | **Threading Building Blocks**, Intel's task-parallel C++ library | Uses many CPU cores |

OpenMP and TBB are **CPU parallelism libraries**: they spread work across your processor's
cores, the CPU-side counterpart of what CUDA does on the GPU. With the default CPP backend,
host-side execution is serial (one core). Select OMP or TBB and the same Thrust call running
on the host instead uses all your cores.

You do not choose the backend at the call site; it is fixed at **compile time** by defining a
macro. For example, to make the host backend OpenMP:

```bash
nvcc -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -Xcompiler -fopenmp myprog.cu ...
```

This is why `thrust::par` (parallel) and `thrust::seq` (sequential) exist as policies: they
express intent that a multi-threaded host backend like OMP or TBB can act on. On the default
CPP backend there is nothing to parallelize across, so the distinction has no visible effect.
The examples in this tutorial use the defaults — CUDA on the device, serial CPP on the host —
so no extra flags are needed.


```python
%%writefile thrust_exec_policy.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <iostream>

int main() {
    const int N = 10;

    // Host data: ask Thrust to run the reduction on the CPU.
    thrust::host_vector<int> H(N);
    thrust::sequence(H.begin(), H.end(), 1);              // 1..10
    int host_sum = thrust::reduce(thrust::host, H.begin(), H.end());

    // Device data: ask Thrust to run the reduction on the GPU.
    thrust::device_vector<int> D(N);
    thrust::sequence(D.begin(), D.end(), 1);              // 1..10
    int device_sum = thrust::reduce(thrust::device, D.begin(), D.end());

    std::cout << "thrust::host   reduce (runs on CPU): " << host_sum   << std::endl;
    std::cout << "thrust::device reduce (runs on GPU): " << device_sum << std::endl;
    return 0;
}

```

    Overwriting thrust_exec_policy.cu



```python
!nvcc -arch=sm_89 thrust_exec_policy.cu -o thrust_exec_policy
!./thrust_exec_policy
```

    thrust::host   reduce (runs on CPU): 55
    thrust::device reduce (runs on GPU): 55


---

## 3. Combining Algorithms: SAXPY Example

### What is SAXPY?

**SAXPY** stands for: **S**ingle-precision **A** times **X** Plus **Y**

It's a fundamental operation in linear algebra: **Y = a*X + Y**

Where:
- `a` is a scalar (single number)
- `X` and `Y` are vectors (arrays of numbers)
- The operation updates `Y` by adding `a` times each element of `X`

**Example:**
```
a = 3
X = [1, 2, 3, 4, 5]
Y = [2, 2, 2, 2, 2]

Result: Y = [5, 8, 11, 14, 17]
         (because: [3*1+2, 3*2+2, 3*3+2, 3*4+2, 3*5+2])
```

---

### What is BLAS?

**BLAS** = **B**asic **L**inear **A**lgebra **S**ubprograms

It's a **standard library** of fundamental linear algebra operations used in:
- Scientific computing
- Machine learning (neural networks use matrix operations extensively)
- Computer graphics
- Physics simulations
- Signal processing

BLAS operations are highly optimized and serve as building blocks for more complex algorithms. SAXPY is one of the simplest and most commonly used BLAS operations.

**Why learn SAXPY?**
- It's simple enough to understand completely
- Complex enough to demonstrate combining Thrust algorithms
- Used in real-world applications (gradient descent in ML, physics simulations, etc.)
- Shows how functors can store state (`a`) and use it in calculations

---

### SAXPY Implementation with Thrust

This demonstrates how to combine Thrust algorithms to implement common numerical operations.
The work is done by a single **binary** `transform` — the form that reads from *two* input
ranges and writes one output:

```cpp
// Y = a*X + Y, computed directly inside main()
thrust::transform(X.begin(), X.end(),  // input 1 -> the functor's x
                  Y.begin(),           // input 2 -> the functor's y
                  Y.begin(),           // output  -> written back into Y
                  saxpy_functor(a));   // op(x, y) = a*x + y
```

The four iterator arguments are: input range 1 (`X.begin()`, `X.end()`), the start of input
range 2 (`Y.begin()`), the start of the output (`Y.begin()` again), and the binary operation.
For each index `i`, Thrust computes `op(X[i], Y[i])` and stores it — so
`Y[i] = a*X[i] + Y[i]`. The scalar `a` rides inside the functor as captured state (`a*x + y`),
which is why a *functor* is used here rather than a built-in like `thrust::plus`: the built-ins
combine two values, but none of them multiplies one input by a stored constant first.

**Why writing into `Y` is safe.** Notice the output iterator (`Y.begin()`) is the *same* as
input range 2. Reusing an input range as the output is an **in-place transformation**, and
Thrust explicitly permits it — the `transform` documentation states "the input and output
sequences may coincide, resulting in an in-place transformation." It is safe because
`transform` is strictly element-wise: producing `Y[i]` reads only `X[i]` and `Y[i]`, never a
neighbor, so overwriting `Y[i]` cannot corrupt another element's input. (`X` is a separate
buffer and is left unchanged.) This is exactly what SAXPY wants — it updates `Y` in place
rather than allocating a third vector for the result.


```python
%%writefile thrust_saxpy.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <iostream>

// Functor for SAXPY: returns a*x + y for one pair of elements
struct saxpy_functor {
    const float a;

    saxpy_functor(float _a) : a(_a) {}  //constructor

    __host__ __device__
    float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

int main() {
    const int N = 5;

    // Initialize vectors
    thrust::device_vector<float> X(N);
    thrust::device_vector<float> Y(N);

    thrust::sequence(X.begin(), X.end(), 1.0f);  // X = [1, 2, 3, 4, 5]
    thrust::fill(Y.begin(), Y.end(), 2.0f);      // Y = [2, 2, 2, 2, 2]

    float a = 3.0f;

    // Print initial values
    thrust::host_vector<float> H_X = X;
    thrust::host_vector<float> H_Y = Y;

    std::cout << "Before SAXPY:" << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "X = ";
    for(int i = 0; i < N; i++) std::cout << H_X[i] << " ";
    std::cout << std::endl << "Y = ";
    for(int i = 0; i < N; i++) std::cout << H_Y[i] << " ";
    std::cout << std::endl;

    // Perform SAXPY: Y = a*X + Y
    // transform feeds each (X[i], Y[i]) pair to the functor and writes the result back into Y.
    thrust::transform(X.begin(), X.end(),  // input 1 -> functor's x
                      Y.begin(),           // input 2 -> functor's y
                      Y.begin(),           // output  -> written back into Y (in place)
                      saxpy_functor(a));   // op(x, y) = a*x + y

    // Print result
    H_Y = Y;
    std::cout << "\nAfter SAXPY (Y = a*X + Y):" << std::endl;
    std::cout << "Y = ";
    for(int i = 0; i < N; i++) std::cout << H_Y[i] << " ";
    std::cout << std::endl;

    std::cout << "\nExpected: [5, 8, 11, 14, 17]" << std::endl;

    return 0;
}

```

    Overwriting thrust_saxpy.cu



```python
!nvcc -arch=sm_89 thrust_saxpy.cu -o thrust_saxpy
!./thrust_saxpy
```

    Before SAXPY:
    a = 3
    X = 1 2 3 4 5 
    Y = 2 2 2 2 2 
    
    After SAXPY (Y = a*X + Y):
    Y = 5 8 11 14 17 
    
    Expected: [5, 8, 11, 14, 17]


---

## 4. More Algorithms

### 4.1 Copy

**`thrust::copy`** copies elements from one range to another.


```python
%%writefile thrust_copy.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <iostream>

int main() {
    thrust::device_vector<int> source(5);
    thrust::sequence(source.begin(), source.end(), 10);  // 10, 11, 12, 13, 14
    
    thrust::device_vector<int> dest(5, 0);  // Initialize with zeros
    
    // Copy from source to dest
    thrust::copy(source.begin(), source.end(), dest.begin());
    
    thrust::host_vector<int> H = dest;
    std::cout << "Copied values: ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

    Overwriting thrust_copy.cu



```python
!nvcc -arch=sm_89 thrust_copy.cu -o thrust_copy
!./thrust_copy
```

    Copied values: 10 11 12 13 14 


### 4.2 Sorting

Thrust provides highly optimized sorting algorithms:
- **`thrust::sort`**: In-place sort
- **`thrust::sort_by_key`**: Sort keys and rearrange values accordingly

By default `thrust::sort(D.begin(), D.end())` sorts **ascending**. To sort **descending**, pass
a third argument — a **comparator**:

```cpp
thrust::sort(D.begin(), D.end(), cuda::std::greater<int>());
```

#### What the comparator does

`sort` has no built-in notion of "ascending" — this is the same algorithm/operation split seen
with `transform` and `reduce`. The **algorithm** moves elements around; the **comparator** only
answers "should `a` come before `b`?" by returning a `bool`. `cuda::std::greater<int>` returns
`a > b`, so `sort` arranges the data largest-first → descending. With no comparator, `sort`
uses `cuda::std::less` (`a < b`) → ascending. You can pass any functor returning `bool` to sort
by a custom rule (by absolute value, by a struct field, and so on).

#### Reading the syntax `cuda::std::greater<int>()`

This one token packs four pieces together:

```text
cuda::std :: greater < int > ()
└────┬────┘   └──┬──┘ └─┬─┘ └┬┘
     1           2      3    4
```

1. **`cuda::std::`** — namespace qualifier. `::` is the scope-resolution operator; it says "the
   `greater` from the `cuda::std` namespace" (CCCL's CUDA-compatible version of the standard
   library, usable on both host and device). This replaces the older `thrust::greater`, which
   still works but is **deprecated** in current CUDA toolkits.
2. **`greater`** — a *class template*, i.e. a blueprint, not yet a usable type. Like
   `std::vector`, it needs a type argument before you can make one.
3. **`<int>`** — the template argument. `greater<int>` is now a concrete **type** whose
   `operator()(int a, int b)` returns `a > b`. It must match the element type of `D`
   (`device_vector<int>`).
4. **`()`** — constructs an **object** of that type. `greater<int>` is a *type*; you cannot pass
   a type to `sort`, you pass a value. The trailing `()` calls the default constructor to make a
   temporary, unnamed comparator object — which is what `sort` receives.

So `cuda::std::greater<int>()` means "create a temporary `greater<int>` object." It is exactly
equivalent to the two-step form:

```cpp
cuda::std::greater<int> cmp;            // make a named object
thrust::sort(D.begin(), D.end(), cmp);  // pass it
```

(If you ever call the comparator directly you will see *two* sets of parentheses —
`cuda::std::greater<int>()(9, 5)` — the first `()` builds the object, the second calls its
`operator()`. `sort` makes that second call for you internally.)

Using it requires `#include <cuda/std/functional>`.


```python
%%writefile thrust_sort.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda/std/functional>   // for cuda::std::greater (the comparator)
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Create random data
    thrust::host_vector<int> H(10);
    srand(time(NULL));
    for(int i = 0; i < 10; i++) {
        H[i] = rand() % 100;
    }
    
    std::cout << "Before sort: ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    // Transfer to device and sort (ascending by default)
    thrust::device_vector<int> D = H;
    thrust::sort(D.begin(), D.end());
    
    // Copy back
    H = D;
    std::cout << "After sort:  ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    // Sort in descending order by passing a comparator object
    thrust::sort(D.begin(), D.end(), cuda::std::greater<int>());
    
    H = D;
    std::cout << "Descending:  ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

```

    Overwriting thrust_sort.cu



```python
!nvcc -arch=sm_89 thrust_sort.cu -o thrust_sort
!./thrust_sort
```

    Before sort: 98 37 68 80 41 49 82 97 44 18 
    After sort:  18 37 41 44 49 68 80 82 97 98 
    Descending:  98 97 82 80 68 49 44 41 37 18 


### 4.3 Prefix Sum (Scan)

**`thrust::inclusive_scan`** and **`thrust::exclusive_scan`** compute **prefix sums** — one of the most fundamental parallel primitives.

---

#### What Is a Prefix Sum?

A **prefix sum** (also called a **scan**) transforms an array so that each output element holds the running total of all previous (and optionally the current) input elements.

Given the all-ones input we use in the code example below (chosen so the running totals are obvious — each output simply counts the elements seen so far):

```
Index:          0  1  2  3  4  5  6  7  8  9
Input:          1  1  1  1  1  1  1  1  1  1

Inclusive:      1  2  3  4  5  6  7  8  9  10   ← position i = i+1 (counts itself)
                ↑  each position = sum of input[0..i]   (inclusive of i)

Exclusive:      0  1  2  3  4  5  6  7  8  9    ← position i = i (excludes itself)
                ↑  each position = sum of input[0..i-1] (exclusive of i)
```

---

#### Inclusive vs Exclusive — the One-Position Shift

The two variants differ by exactly one position. The exclusive output at position `i` equals the inclusive output at position `i-1`, with a leading `0` inserted:

```
Input:          [a,  b,  c,  d,  e]

Inclusive:      [a,  a+b, a+b+c, a+b+c+d, a+b+c+d+e]
                 ↑   ↑    ↑      ↑         ↑
                 i=0 includes itself; the current element is folded in

Exclusive:      [0,  a,   a+b,   a+b+c,   a+b+c+d]
                 ↑   ↑    ↑      ↑         ↑
                 i=0 is always 0 (nothing came before it)
                 each output is the sum of ONLY the elements before it
```

**Concrete example with `[3, 1, 4, 1, 5]`** (varied values make the difference clearer than all-ones):

```
Index:          0   1   2   3   4
Input:          3   1   4   1   5

Inclusive:      3   4   8   9   14
                3 | 3+1 | 3+1+4 | 3+1+4+1 | 3+1+4+1+5

Exclusive:      0   3   4   8    9
                0 | 3   | 3+1   | 3+1+4   | 3+1+4+1
```

---

#### Step-By-Step: How Parallel Scan Works on the GPU

A naive sequential loop computes each output one at a time:

```
for (int i = 1; i < N; i++)
    out[i] = out[i-1] + in[i];   // ← strictly sequential, can't parallelize
```

Thrust uses a work-efficient **parallel scan** algorithm (often called the Blelloch scan):

```
┌─────────────────────────────────────────────────────────────┐
│  PARALLEL SCAN — CONCEPTUAL OVERVIEW                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Upsweep (reduce tree — build partial sums)        │
│                                                             │
│  [1, 1, 1, 1, 1, 1, 1, 1]    ← 8 elements                  │
│   ╲ ╱   ╲ ╱   ╲ ╱   ╲ ╱                                    │
│   [2,    2,    2,    2]       ← add pairs in parallel       │
│     ╲   ╱       ╲   ╱                                       │
│     [4,          4]           ← add pairs in parallel       │
│       ╲          ╱                                          │
│        [8]                    ← total sum                   │
│                                                             │
│  Phase 2: Downsweep (distribute partial sums)               │
│                                                             │
│  Works back down the tree, filling in prefix values.        │
│  Each level runs fully in parallel.                         │
│                                                             │
│  Result: O(log N) steps instead of O(N) — massive speedup  │
│  on large arrays!                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Thrust handles all of this automatically. You just call the function.

---

#### API Syntax

```cpp
// Inclusive scan (output[i] includes input[i])
thrust::inclusive_scan(input.begin(), input.end(), output.begin());

// Exclusive scan (output[i] excludes input[i])
thrust::exclusive_scan(input.begin(), input.end(), output.begin());

// With a custom initial value (exclusive only)
thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 100);
// output[0] = 100 instead of 0

// With a custom binary operator
thrust::inclusive_scan(input.begin(), input.end(), output.begin(),
                       thrust::multiplies<int>());
// prefix products instead of prefix sums
```

**Memory note:** input and output are separate arrays — the input is left unchanged. You *can* scan in place by passing the same vector for both input and output (`thrust::inclusive_scan(D.begin(), D.end(), D.begin())`); Thrust handles the overlap safely.

---

#### Why Is Scan Useful?

Scan sounds abstract, but it unlocks many important parallel algorithms:

**1. Dynamic array compaction (stream compaction)**
```
Problem: Remove zeros from [3, 0, 1, 0, 5, 2]
                                            ↓
Step 1: Flag non-zeros:     [1, 0, 1, 0, 1, 1]
Step 2: Exclusive scan:     [0, 1, 1, 2, 2, 3]  ← output index for each kept element
Step 3: Scatter:            [3, 1, 5, 2]         ← compact result!

Each thread uses its scanned flag as its output address → fully parallel writes.
```

**2. Variable-length output (e.g. raycasting, collision detection)**
```
Each thread produces a different number of results.
Scan turns "how many results does each thread produce?"
into "where should each thread write its results?" — with no conflicts.
```

**3. Parallel histogram and bucketing**
```
Exclusive scan of bucket sizes → start offset of each bucket in the output array.
```

**4. In-place running total**
```cpp
// Rolling average requires knowing the sum of all previous elements
// Exclusive scan gives you that directly for every position at once.
```

---

#### Custom Binary Operator

The default operation is addition, but you can supply any associative binary operator:

```cpp
thrust::device_vector<int> D = {1, 2, 3, 4, 5};
thrust::device_vector<int> D_product(5);

// Prefix products: 1, 1*2, 1*2*3, ...
thrust::inclusive_scan(D.begin(), D.end(), D_product.begin(),
                       thrust::multiplies<int>());
// D_product = [1, 2, 6, 24, 120]

// Prefix max: running maximum
thrust::inclusive_scan(D.begin(), D.end(), D_product.begin(),
                       thrust::maximum<int>());
// D_product = [1, 2, 3, 4, 5]
```

The operator must be **associative** (`a ⊕ (b ⊕ c) == (a ⊕ b) ⊕ c`) for a parallel scan to produce correct results.

---

#### Scan vs Reduce — Same Operator, Different Shape

We used `thrust::maximum<int>()` earlier with **`thrust::reduce`** (Section 2.3) to find the single largest element. That same operator works with `inclusive_scan` — but the **algorithm** decides the shape of the output, not the operator:

```
Input:                 [3,  1,  4,  1,  5]

reduce(max):           5                     ← one final answer
                       (folds N values into 1)

inclusive_scan(max):   [3,  3,  4,  4,  5]   ← every prefix answer
                       (one running max per position, N values out)
```

| | `reduce` + `maximum` | `inclusive_scan` + `maximum` |
|---|---|---|
| **Question** | "What is the largest value?" | "What is the largest value *seen so far* at each position?" |
| **Output** | one scalar | a full array (same length as input) |
| **Shape** | collapses N → 1 | maps N → N |

**The link between them:** the *last* element of an inclusive scan is exactly what `reduce` returns. Scan just keeps every intermediate accumulation on the way there, instead of throwing them away.

```
inclusive_scan(max): [3, 3, 4, 4, 5]
                                  ↑
                     this last element == reduce(max) == 5
```

Use **reduce** when you want *the* answer (the max, the sum, the product). Use **scan** when each position needs the answer *up to that point* — e.g. a running "high-water mark" over a time series, or a peak-so-far in a streaming signal. Reduce can't give you that trail; scan can.

---

#### One-Line Summary

> **Scan turns a sequence into its running totals — inclusive keeps the current element, exclusive shifts right by one and seeds with zero. It runs in O(log N) parallel steps and unlocks dozens of higher-level parallel algorithms.**


```python
%%writefile thrust_scan.cu
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <iostream>

int main() {
    thrust::device_vector<int> D(10);
    thrust::fill(D.begin(), D.end(), 1);  // All elements = 1
    
    thrust::device_vector<int> D_inclusive(10);
    thrust::device_vector<int> D_exclusive(10);
    
    // Inclusive scan
    thrust::inclusive_scan(D.begin(), D.end(), D_inclusive.begin());
    
    // Exclusive scan
    thrust::exclusive_scan(D.begin(), D.end(), D_exclusive.begin());
    
    thrust::host_vector<int> H_inc = D_inclusive;
    thrust::host_vector<int> H_exc = D_exclusive;
    
    std::cout << "Input:          ";
    for(int i = 0; i < 10; i++) std::cout << 1 << " ";
    std::cout << std::endl;
    
    std::cout << "Inclusive scan: ";
    for(int i = 0; i < H_inc.size(); i++) {
        std::cout << H_inc[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Exclusive scan: ";
    for(int i = 0; i < H_exc.size(); i++) {
        std::cout << H_exc[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

    Overwriting thrust_scan.cu



```python
!nvcc -arch=sm_89 thrust_scan.cu -o thrust_scan
!./thrust_scan
```

    Input:          1 1 1 1 1 1 1 1 1 1 
    Inclusive scan: 1 2 3 4 5 6 7 8 9 10 
    Exclusive scan: 0 1 2 3 4 5 6 7 8 9 


## 5. Fancy Iterators

Fancy iterators are lightweight *views*: they produce values on demand instead of reading
them from an allocated array. Because Thrust algorithms work on any iterator, these let you
feed constants, counts, on-the-fly transforms, or several zipped sequences into an algorithm
without materializing a temporary `device_vector`. The result is fewer allocations and less
memory traffic, which is what GPU performance is usually bound by.

### 5.1 Constant Iterator

**`thrust::constant_iterator`** represents an infinite sequence of the same value.

#### What This Does

Add the same number to every element in a vector — without creating an extra array!

```cpp
D = [1, 2, 3, 4, 5]
Add 10 to each element
Result: [11, 12, 13, 14, 15]
```

---

#### Understanding the Problem

**The wasteful way:**
```cpp
thrust::device_vector<int> D = {1, 2, 3, 4, 5};
thrust::device_vector<int> tens = {10, 10, 10, 10, 10};  // Wasteful!
thrust::device_vector<int> result(5);

thrust::transform(D.begin(), D.end(), 
                 tens.begin(),  // Need to create this whole vector!
                 result.begin(), 
                 thrust::plus<int>());
```

```
Memory: [1,2,3,4,5] + [10,10,10,10,10]  ← 2x memory usage!
```

**The smart way using constant_iterator:**
```cpp
thrust::device_vector<int> D = {1, 2, 3, 4, 5};
thrust::constant_iterator<int> const_iter(10);  // "Infinite" 10s!
thrust::device_vector<int> result(5);

thrust::transform(D.begin(), D.end(), 
                 const_iter,    // Just one iterator, no vector needed!
                 result.begin(), 
                 thrust::plus<int>());
```

```
Memory: [1,2,3,4,5]  ← Only original data!
```

---

#### How `constant_iterator` Works

A `constant_iterator` is like a **magical iterator that always returns the same value** no matter how many times you read from it:

```cpp
thrust::constant_iterator<int> const_iter(10);

// Reading from it:
*const_iter       // Returns 10
*(const_iter + 1) // Returns 10
*(const_iter + 2) // Returns 10
*(const_iter + 100) // Still returns 10!
```

```
┌─────────────────────────────────────────────────────────────┐
│              CONSTANT ITERATOR CONCEPT                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   const_iter(10):  [10, 10, 10, 10, 10, 10, ...]          │
│                     ↑   ↑   ↑   ↑   ↑   ↑                  │
│                     └───┴───┴───┴───┴───┴───── All the same!│
│                                                             │
│   It's "infinite" - you can advance it as far as you want  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Breaking Down the Transform Call

This is a **binary transform** (takes two inputs, produces one output):

```cpp
thrust::transform(D.begin(), D.end(),    // Input 1: [1, 2, 3, 4, 5]
                 const_iter,             // Input 2: [10, 10, 10, 10, 10, ...]
                 result.begin(),         // Output: where to store results
                 thrust::plus<int>());   // Operation: add the two inputs
```

**What happens step by step:**

```
Step 1: Read D[0]=1, read const_iter[0]=10, compute 1+10=11, write to result[0]
Step 2: Read D[1]=2, read const_iter[1]=10, compute 2+10=12, write to result[1]
Step 3: Read D[2]=3, read const_iter[2]=10, compute 3+10=13, write to result[2]
Step 4: Read D[3]=4, read const_iter[3]=10, compute 4+10=14, write to result[3]
Step 5: Read D[4]=5, read const_iter[4]=10, compute 5+10=15, write to result[5]
```

**Visual representation:**

```
Input 1 (D):          [1,  2,  3,  4,  5]
                       |   |   |   |   |
                       +   +   +   +   +  ← thrust::plus
                       |   |   |   |   |
Input 2 (const_iter): [10, 10, 10, 10, 10] (infinite sequence of 10s)
                       |   |   |   |   |
                       ↓   ↓   ↓   ↓   ↓
Output (result):      [11, 12, 13, 14, 15]
```

---

#### Simple Analogy

```
┌─────────────────────────────────────────────────────────────┐
│  COPY MACHINE ANALOGY                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   constant_iterator(10) = Copy machine with infinite paper │
│                           that always prints "10"           │
│                                                             │
│   You: "Give me the 1st copy"  →  Machine: "10"           │
│   You: "Give me the 2nd copy"  →  Machine: "10"           │
│   You: "Give me the 100th copy" → Machine: "10"           │
│                                                             │
│   It never runs out, always returns the same thing!         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Why This is Better

**Comparison:**

| Without constant_iterator | With constant_iterator |
|--------------------------|------------------------|
| Create vector: `thrust::device_vector<int> tens(5, 10);` | Create iterator: `thrust::constant_iterator<int> const_iter(10);` |
| Allocates memory for 5 integers | No memory allocation! |
| Transfer data to GPU | No transfer needed! |
| Use in transform | Use in transform |

---

#### Complete Example Walkthrough

Let's trace through the full example in our code:

```cpp
thrust::device_vector<int> D(5);
thrust::sequence(D.begin(), D.end(), 1);  // D = [1, 2, 3, 4, 5]

thrust::constant_iterator<int> const_iter(10);  // Infinite 10s

thrust::device_vector<int> result(5);  // Storage for output

// Binary transform: add corresponding elements
thrust::transform(D.begin(), D.end(),  // Range from D
                 const_iter,           // Start of constant 10s
                 result.begin(),       // Where to store output
                 thrust::plus<int>()); // Add operation

// result is now [11, 12, 13, 14, 15]
```

**Why it works:**
- `thrust::transform` reads pairs of elements: `(D[i], const_iter[i])`
- For each pair, it applies `thrust::plus` which adds them
- The constant iterator always provides 10, no matter what index
- Results get written to the output vector

---

#### Common Use Cases

```cpp
// Add constant to all elements
thrust::transform(D.begin(), D.end(), 
                 thrust::constant_iterator<int>(10),
                 result.begin(), 
                 thrust::plus<int>());

// Subtract constant from all elements
thrust::transform(D.begin(), D.end(), 
                 thrust::constant_iterator<int>(10),
                 result.begin(), 
                 thrust::minus<int>());

// Multiply all elements by constant
thrust::transform(D.begin(), D.end(), 
                 thrust::constant_iterator<float>(3.14),
                 result.begin(), 
                 thrust::multiplies<float>());
```

---

#### One-Line Summary

> **Constant iterators provide an infinite sequence of the same value without allocating memory — perfect for operations with scalars.**

---


```python
%%writefile thrust_constant_iterator.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <iostream>

int main() {
    thrust::device_vector<int> D(5);
    thrust::sequence(D.begin(), D.end(), 1);  // [1, 2, 3, 4, 5]
    
    // Add 10 to all elements using constant_iterator
    thrust::constant_iterator<int> const_iter(10);
    
    thrust::device_vector<int> result(5);
    thrust::device_vector<int> result2(5);
    thrust::transform(D.begin(), D.end(), 
                     const_iter, 
                     result.begin(), 
                     thrust::plus<int>());

    thrust::transform(D.begin(), D.end(), 
                     const_iter, 
                     result2.begin(), 
                     thrust::minus<int>());

    thrust::host_vector<int> H = result;
    std::cout << "D + 10 = ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;

    H = result2;
    std::cout << "D - 10 = ";
    for(int i = 0; i < H.size(); i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

    Overwriting thrust_constant_iterator.cu



```python
!nvcc -arch=sm_89 thrust_constant_iterator.cu -o thrust_constant_iterator
!./thrust_constant_iterator
```

    D + 10 = 11 12 13 14 15 
    D - 10 = -9 -8 -7 -6 -5 


### 5.2 Counting Iterator

**`thrust::counting_iterator`** represents a sequence of incrementing values — like `1, 2, 3, 4, ...` — without storing them in memory!

#### What This Does

Compute `1 + 2 + 3 + ... + 10 = 55` without creating an array!

---

#### Understanding the Problem

**The traditional way:**
```cpp
thrust::device_vector<int> numbers(10);
thrust::sequence(numbers.begin(), numbers.end(), 1);  // [1,2,3,...,10]
int sum = thrust::reduce(numbers.begin(), numbers.end());
```

```
GPU Memory: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  ← Allocates 40 bytes
                                                 (10 ints × 4 bytes)
```

**The smart way with counting_iterator:**
```cpp
thrust::counting_iterator<int> first(1);   // Starts at 1
thrust::counting_iterator<int> last(11);   // Stops before 11
int sum = thrust::reduce(first, last);
```

```
Memory: NOTHING! ← No allocation needed!
```

---

#### How `counting_iterator` Works

A `counting_iterator` is like a **virtual sequence generator** — it computes values on-the-fly:

```cpp
thrust::counting_iterator<int> iter(1);  // Start at 1

// Reading from it:
*iter        // Returns 1
*(iter + 1)  // Returns 2
*(iter + 2)  // Returns 3
*(iter + 5)  // Returns 6
*(iter + 99) // Returns 100
```

```
┌─────────────────────────────────────────────────────────────┐
│            COUNTING ITERATOR CONCEPT                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   counting_iterator(1):  [1, 2, 3, 4, 5, 6, 7, ...]       │
│                           ↑  ↑  ↑  ↑  ↑  ↑  ↑              │
│                           │  │  │  │  │  │  │              │
│                      start + 0, 1, 2, 3, 4, 5, 6...        │
│                                                             │
│   Each position returns: start_value + position            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Simple Analogy

```
┌─────────────────────────────────────────────────────────────┐
│  ODOMETER ANALOGY                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   counting_iterator = Car odometer                          │
│                                                             │
│   Set to start at mile 1                                    │
│   Every time you "drive" (advance the iterator):            │
│     Mile 1 → Mile 2 → Mile 3 → Mile 4 ...                  │
│                                                             │
│   The odometer doesn't store all the numbers!               │
│   It just calculates the current mile when you look.        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Understanding the Range

```cpp
thrust::counting_iterator<int> first(1);    // Start at 1
thrust::counting_iterator<int> last(11);    // Stop before 11
```

C++ uses **half-open ranges: [first, last)** — includes first, excludes last:

```
    first(1)                                      last(11)
       │                                             │
       ▼                                             ▼
     [1,  2,  3,  4,  5,  6,  7,  8,  9,  10]      (11 not included)
      ↑                                    ↑
   included                            included
```

**Why `last = first + 10`?**

```cpp
first(1)  → returns values: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
last = first + 10  
      = counting_iterator(1) + 10
      = counting_iterator(11)  ← One past the last value we want
```

---

#### What Happens Inside `reduce()`

```cpp
int sum = thrust::reduce(first, last);
```

```
┌────────────────────────────────────────────────────────────┐
│  STEP BY STEP                                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  reduce() asks first iterator: "give me value at position 0"│
│    → counting_iterator computes: 1 + 0 = 1                │
│    → Partial sum: 0 + 1 = 1                               │
│                                                            │
│  reduce() asks: "give me value at position 1"              │
│    → counting_iterator computes: 1 + 1 = 2                │
│    → Partial sum: 1 + 2 = 3                               │
│                                                            │
│  reduce() asks: "give me value at position 2"              │
│    → counting_iterator computes: 1 + 2 = 3                │
│    → Partial sum: 3 + 3 = 6                               │
│                                                            │
│  ... continues until position 9 ...                        │
│                                                            │
│  reduce() asks: "give me value at position 9"              │
│    → counting_iterator computes: 1 + 9 = 10               │
│    → Partial sum: 45 + 10 = 55                            │
│                                                            │
│  Result: 55                                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

#### Memory Comparison

```
┌─────────────────────────────────────────────────────────────┐
│  TRADITIONAL APPROACH                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GPU Memory:                                                │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬────┐               │
│  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ 10 │  ← 40 bytes   │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴────┘               │
│                                                             │
│  Operations:                                                │
│   1. Allocate 40 bytes on GPU                               │
│   2. Generate sequence on GPU                               │
│   3. Read all values during reduce                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  COUNTING ITERATOR APPROACH                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GPU Memory:                                                │
│  [EMPTY]                                   ← 0 bytes!       │
│                                                             │
│  Operations:                                                │
│   1. Create iterator with start=1                           │
│   2. Compute values on-the-fly during reduce                │
│   3. No memory allocation or transfers                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Common Use Cases

**1. Generate indices:**
```cpp
// Get indices 0, 1, 2, 3, 4
thrust::counting_iterator<int> first(0);
thrust::counting_iterator<int> last(5);
```

**2. Sum of first N numbers:**
```cpp
// Sum 1 + 2 + ... + 1000
thrust::counting_iterator<int> first(1);
int sum = thrust::reduce(first, first + 1000);
```

**3. With other algorithms:**
```cpp
// Copy sequence into vector
thrust::counting_iterator<int> first(100);  // Start at 100
thrust::counting_iterator<int> last(110);   // [100..109]
thrust::device_vector<int> vec(10);
thrust::copy(first, last, vec.begin());  // vec = [100,101,...,109]
```

**4. Custom step sizes (using transform_iterator):**
```cpp
// Generate even numbers: 0, 2, 4, 6, 8
struct times_two {
    __host__ __device__
    int operator()(int x) const { return x * 2; }
};

auto first = thrust::make_transform_iterator(
    thrust::counting_iterator<int>(0), times_two());
// first gives: 0, 2, 4, 6, 8, ...
```

---

#### Why This is Powerful

For large sequences, the savings are dramatic:

```
Compute sum of first 1,000,000 numbers:

Traditional:  4 MB allocated
              Transfer to GPU
              Read from memory during reduce

counting_iterator:  0 bytes allocated
                    No transfer
                    Values computed in GPU registers
```

**Memory bandwidth matters on GPUs!** Avoiding unnecessary memory operations can significantly speed up your code.

---

#### One-Line Summary

> **Counting iterators generate sequences on-the-fly without memory allocation — perfect for indices and simple sequences.**

---


```python
%%writefile thrust_counting_iterator.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <iostream>

int main() {
    // Sum numbers from 1 to 10 without creating a vector
    thrust::counting_iterator<int> first(1);
    thrust::counting_iterator<int> last = first + 10;
    
    int sum = thrust::reduce(first, last);
    
    std::cout << "Sum of 1 to 10 (using counting_iterator): " << sum << std::endl;
    std::cout << "Expected: 55" << std::endl;
    
    return 0;
}
```

    Overwriting thrust_counting_iterator.cu



```python
!nvcc -arch=sm_89 thrust_counting_iterator.cu -o thrust_counting_iterator
!./thrust_counting_iterator
```

    Sum of 1 to 10 (using counting_iterator): 55
    Expected: 55


### 5.3 Transform Iterator

**`thrust::transform_iterator`** applies a function on-the-fly when accessing elements, without creating intermediate arrays.

#### What This Does

Computes **1² + 2² + 3² + 4² + 5² = 55** on a GPU — efficiently!

---

#### Understanding the Problem

```cpp
// Traditional wasteful way:
thrust::device_vector<float> D = {1, 2, 3, 4, 5};
thrust::device_vector<float> D_squared(5);  // Creates extra array!
thrust::transform(D.begin(), D.end(), D_squared.begin(), square());
float sum = thrust::reduce(D_squared.begin(), D_squared.end());

// Smart way with transform_iterator:
auto first = thrust::make_transform_iterator(D.begin(), square());
auto last = thrust::make_transform_iterator(D.end(), square());
float sum = thrust::reduce(first, last);  // No extra array!
```

**Memory comparison:**
```
┌─────────────────────────────────────────────────────────────┐
│  WITHOUT transform iterators (wasteful):                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Step 1: Create squared array                              │
│   Memory: [1,2,3,4,5] + [1,4,9,16,25]  ← 2x memory!        │
│                                                             │
│   Step 2: Sum the squared array                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WITH transform iterators (efficient):                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Memory: [1,2,3,4,5]  ← Only original data!               │
│                                                             │
│   Squaring happens during reading, never stored.            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Wait, Why Two Iterators?

Great question! `first` and `last` are **not** two separate arrays — they're just **position markers** (like bookmarks) that define where to start and stop.

```
                    first                              last
                      │                                  │
                      ▼                                  ▼
                    ┌─────┬─────┬─────┬─────┬─────┐     (one past end)
   GPU Data:        │  1  │  2  │  3  │  4  │  5  │
                    └─────┴─────┴─────┴─────┴─────┘
                      ↑                          ↑
                   D.begin()                  D.end()
```

C++ always defines ranges as **[start, end)** — you need TWO positions:

```
┌────────────────────────────────────────────────────┐
│   thrust::reduce(first, last)                      │
│                    │      │                        │
│                    │      └── "stop HERE"          │
│                    └──────── "start HERE"          │
└────────────────────────────────────────────────────┘
```

**Book analogy:**
```
"Read pages 10-50 and summarize"
          │   │
          │   └── ending position (last)
          └────── starting position (first)

You need BOTH positions to define what to read!
But there's still only ONE book.
```

---

#### What's Inside a Transform Iterator?

```cpp
auto first = thrust::make_transform_iterator(D.begin(), square());
```

```
┌─────────────────────────────────────────┐
│         Transform Iterator              │
│              "first"                    │
├─────────────────────────────────────────┤
│                                         │
│   ┌─────────────────────────────────┐   │
│   │  Original iterator: D.begin() ──────────► Points to D
│   └─────────────────────────────────┘   │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │  Function to apply: square()    │   │
│   └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

**The iterator bundles TWO things together:**
1. WHERE to find the data (`D.begin()`)
2. WHAT to do with it (`square()`)

---

#### How Does `reduce()` Know Which Data to Use?

There is **NO squared vector** in memory! The transform iterator carries a **reference** to the original data inside it.

When `reduce()` asks for values:

```
┌────────────────────────────────────────────────────────────────┐
│  STEP BY STEP                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. reduce() asks first iterator: "give me a value"            │
│                         │                                      │
│                         ▼                                      │
│  2. first iterator:                                            │
│     - Goes to D.begin() (the original vector)                  │
│     - Reads value: 1                                           │
│     - Applies square(): 1² = 1                                 │
│     - Returns: 1                                               │
│                         │                                      │
│                         ▼                                      │
│  3. reduce() asks: "give me next value"                        │
│                         │                                      │
│                         ▼                                      │
│  4. first iterator:                                            │
│     - Goes to D[1]                                             │
│     - Reads value: 2                                           │
│     - Applies square(): 2² = 4                                 │
│     - Returns: 4                                               │
│                         │                                      │
│                         ▼                                      │
│  ... and so on until it reaches "last"                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Simple analogy:**
```
┌─────────────────────────────────────────────────────────────┐
│   TRANSLATOR ANALOGY                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   French Book         Translator           You              │
│   (original D)        (transform           (reduce)         │
│                        iterator)                            │
│                                                             │
│   "Bonjour"  ───────►  reads &   ───────► "Hello"          │
│                        translates                           │
│                                                             │
│   The translator doesn't create a new English book!         │
│   They translate on-the-fly as you ask for each word.       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Complete Visual Timeline

```
GPU MEMORY:
                    ┌─────┬─────┬─────┬─────┬─────┐
  Only this exists: │  1  │  2  │  3  │  4  │  5  │   Vector D
                    └─────┴─────┴─────┴─────┴─────┘
                      ▲
                      │
              first points here
              (with square() attached)
                      │
                      ▼
                ┌───────────┐
                │  square() │    Applied on-the-fly,
                └───────────┘    NOT stored anywhere!
                      │
                      ▼
                    1, 4, 9, 16, 25    ← reduce() sees these
                                         but they're never
                                         stored in memory!
```

---

#### Why Not Just Use `D.begin()` and `D.end()`?

You absolutely **can** — but you'd get a different result!

```cpp
// Using regular iterators
float sum = thrust::reduce(D.begin(), D.end());
// Result: 1 + 2 + 3 + 4 + 5 = 15 ← Just adds raw values!

// Using transform iterators
float sum = thrust::reduce(first, last);
// Result: 1² + 2² + 3² + 4² + 5² = 55 ← Squares first!
```

**What each iterator "sees":**

```
┌─────────────────────────────────────────────────────────────┐
│  REGULAR ITERATORS: D.begin() / D.end()                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   D.begin()                           D.end()               │
│      │                                   │                  │
│      ▼                                   ▼                  │
│    ┌─────┬─────┬─────┬─────┬─────┐                         │
│    │  1  │  2  │  3  │  4  │  5  │  ← Raw values           │
│    └─────┴─────┴─────┴─────┴─────┘                         │
│                                                             │
│    reduce() sees: 1, 2, 3, 4, 5                            │
│    Result: 15                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  TRANSFORM ITERATORS: first / last                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   first                                 last                │
│      │                                   │                  │
│      ▼                                   ▼                  │
│    ┌─────┬─────┬─────┬─────┬─────┐                         │
│    │  1  │  2  │  3  │  4  │  5  │  ← Raw values           │
│    └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                         │
│       │     │     │     │     │                             │
│       ▼     ▼     ▼     ▼     ▼                             │
│      x²    x²    x²    x²    x²    ← Transform applied      │
│       │     │     │     │     │                             │
│       ▼     ▼     ▼     ▼     ▼                             │
│    ┌─────┬─────┬─────┬─────┬─────┐                         │
│    │  1  │  4  │  9  │ 16  │ 25  │  ← What reduce() sees   │
│    └─────┴─────┴─────┴─────┴─────┘                         │
│                                                             │
│    reduce() sees: 1, 4, 9, 16, 25                          │
│    Result: 55                                               │
└─────────────────────────────────────────────────────────────┘
```

**Simple comparison:**
```
┌─────────────────────────────────────────────────────────┐
│  Regular glasses    vs    Sunglasses                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  👓 D.begin()             🕶️ transform_iterator         │
│  See things as-is         See things transformed        │
│                                                         │
│  [1, 2, 3, 4, 5]          [1, 4, 9, 16, 25]            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

#### One-Line Summary

> **Transform iterators let you process data on-the-fly without creating intermediate copies — saving memory and time on the GPU.**

---

#### Quick Reference

| Concept | What It Does |
|---------|-------------|
| `make_transform_iterator` | Wraps iterator + adds transformation |
| `first` / `last` | Position markers that transform when read |
| `reduce(first, last)` | Reads and transforms on-the-fly, never stores intermediate results |
| Memory savings | No temporary arrays created |

---


```python
%%writefile thrust_transform_iterator.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <iostream>

// Square functor
struct square {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

int main() {
    thrust::device_vector<float> D(5);
    thrust::sequence(D.begin(), D.end(), 1.0f);  // [1, 2, 3, 4, 5]
    
    // Compute sum of squares without creating intermediate array
    auto first = thrust::make_transform_iterator(D.begin(), square());
    auto last = thrust::make_transform_iterator(D.end(), square());
    
    float sum_of_squares = thrust::reduce(first, last);
    
    std::cout << "Sum of squares (1^2 + 2^2 + 3^2 + 4^2 + 5^2): " 
              << sum_of_squares << std::endl;
    std::cout << "Expected: 55" << std::endl;
    
    return 0;
}
```

    Overwriting thrust_transform_iterator.cu



```python
!nvcc -arch=sm_89 thrust_transform_iterator.cu -o thrust_transform_iterator
!./thrust_transform_iterator
```

    Sum of squares (1^2 + 2^2 + 3^2 + 4^2 + 5^2): 55
    Expected: 55


### 5.4 Zip Iterator

**`thrust::zip_iterator`** lets one iterator position refer to several sequences at the same time. Each dereference produces a tuple containing one element from each input range.

This is useful when the operation for element `i` needs values from multiple arrays:

```cpp
X = [1, 2, 3, 4, 5]
Y = [2, 2, 2, 2, 2]

zip(X, Y) behaves like:
[(1,2), (2,2), (3,2), (4,2), (5,2)]
```

The example below computes a dot product in two visible stages:

1. Zip `X` and `Y` so each parallel operation receives `(x_i, y_i)` together.
2. Use `thrust::transform` to compute the products `x_i * y_i`.
3. Use `thrust::reduce` to sum those products.

In production Thrust code, this pattern is often written as `thrust::transform_reduce`, which fuses the transform and reduction into one algorithm call. This tutorial keeps the steps separate so you can see how zip iterators, functors, and reductions compose.

The important idea is that the zip iterator does not create a new array of tuples in GPU memory. It is a lightweight view over existing arrays.



```python
%%writefile thrust_zip_iterator.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <iostream>

// Functor to compute dot product contribution: x * y
struct dot_product_functor {
    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t) const {
        return thrust::get<0>(t) * thrust::get<1>(t);
    }
};

int main() {
    const int N = 5;
    
    thrust::device_vector<float> X(N);
    thrust::device_vector<float> Y(N);
    
    thrust::sequence(X.begin(), X.end(), 1.0f);  // [1, 2, 3, 4, 5]
    thrust::fill(Y.begin(), Y.end(), 2.0f);      // [2, 2, 2, 2, 2]
    
    // Create zip_iterator
    auto first = thrust::make_zip_iterator(thrust::make_tuple(X.begin(), Y.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(X.end(), Y.end()));
    
    // Transform to get element-wise products
    thrust::device_vector<float> products(N);
    thrust::transform(first, last, products.begin(), dot_product_functor());
    
    // Display products
    thrust::host_vector<float> H = products;
    std::cout << "Element-wise products: ";
    for(int i = 0; i < N; i++) {
        std::cout << H[i] << " ";
    }
    std::cout << std::endl;
    
    // Compute dot product using reduce
    float dot_product = thrust::reduce(products.begin(), products.end());
    std::cout << "Dot product: " << dot_product << std::endl;
    std::cout << "Expected: 30 (1*2 + 2*2 + 3*2 + 4*2 + 5*2)" << std::endl;
    
    return 0;
}
```

    Overwriting thrust_zip_iterator.cu



```python
!nvcc -arch=sm_89 thrust_zip_iterator.cu -o thrust_zip_iterator
!./thrust_zip_iterator
```

    Element-wise products: 2 4 6 8 10 
    Dot product: 30
    Expected: 30 (1*2 + 2*2 + 3*2 + 4*2 + 5*2)


### 5.5 Interop: Using Thrust with Your Own Kernels

Thrust is not all-or-nothing. A `device_vector` owns a normal block of GPU memory, and
`thrust::raw_pointer_cast(D.data())` hands you the raw `T*` for it. You can pass that pointer
straight into a hand-written `__global__` kernel, then keep using the `device_vector`
afterward. This is the bridge between high-level Thrust code and the custom kernels you write
when an algorithm needs control Thrust does not expose (shared memory, specific launch
configurations, irregular access patterns).

The reverse direction exists too: `thrust::device_ptr<T>` wraps a raw pointer you already
have (e.g. from `cudaMalloc`) so Thrust algorithms can operate on it. Ownership does not
change — `raw_pointer_cast` does not copy or free anything; it just exposes the address.


```python
%%writefile thrust_raw_pointer.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

// A hand-written CUDA kernel Thrust does not provide: multiply each element by 3.
__global__ void triple(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 3.0f;
}

int main() {
    const int N = 5;
    thrust::device_vector<float> D(N);
    for (int i = 0; i < N; i++) D[i] = i + 1;            // 1, 2, 3, 4, 5

    // Hand the device buffer to the raw kernel via raw_pointer_cast.
    float* raw = thrust::raw_pointer_cast(D.data());
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    triple<<<blocks, threads>>>(raw, N);
    cudaDeviceSynchronize();

    // The device_vector still owns the memory; copy the result back to inspect it.
    thrust::host_vector<float> H = D;
    std::cout << "After custom kernel (x3): ";
    for (int i = 0; i < N; i++) std::cout << H[i] << " ";
    std::cout << std::endl;
    return 0;
}

```

    Overwriting thrust_raw_pointer.cu



```python
!nvcc -arch=sm_89 thrust_raw_pointer.cu -o thrust_raw_pointer
!./thrust_raw_pointer
```

    After custom kernel (x3): 3 6 9 12 15 


---

## 6. Advanced Example: Norm Calculation

Now we combine the earlier ideas into a common numerical operation: the L2 norm, also called the Euclidean norm.

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

For the vector `[1, 2, 3, 4]`, the computation is:

```text
sqrt(1*1 + 2*2 + 3*3 + 4*4) = sqrt(30) = 5.47723
```

The direct but less efficient approach would be:

1. Allocate a second vector for squared values.
2. Square each element into that vector.
3. Reduce the squared vector.
4. Take the square root.

The Thrust version below avoids the temporary squared vector. A `transform_iterator` makes `reduce` see squared values on demand:

```cpp
vec[i]                  -> stored in GPU memory
square(vec[i])          -> computed as reduce reads each value
sum of squares          -> produced by thrust::reduce
std::sqrt(sum_of_squares) -> final scalar on the CPU
```

This is the core Thrust style: compose small parallel building blocks, keep large arrays on the GPU, and avoid intermediate storage when an iterator view can express the same computation.



```python
%%writefile thrust_norm.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <iostream>
#include <cmath>

struct square {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

float compute_norm(const thrust::device_vector<float>& vec) {
    // Create transform_iterator that squares each element
    auto first = thrust::make_transform_iterator(vec.begin(), square());
    auto last = thrust::make_transform_iterator(vec.end(), square());
    
    // Sum of squares
    float sum_of_squares = thrust::reduce(first, last);
    
    // Return square root
    return std::sqrt(sum_of_squares);
}

int main() {
    thrust::device_vector<float> vec(4);
    vec[0] = 1.0f;
    vec[1] = 2.0f;
    vec[2] = 3.0f;
    vec[3] = 4.0f;
    
    float norm = compute_norm(vec);
    
    std::cout << "Vector: [1, 2, 3, 4]" << std::endl;
    std::cout << "L2 norm: " << norm << std::endl;
    std::cout << "Expected: " << std::sqrt(1*1 + 2*2 + 3*3 + 4*4) << std::endl;
    
    return 0;
}
```

    Overwriting thrust_norm.cu



```python
!nvcc -arch=sm_89 thrust_norm.cu -o thrust_norm
!./thrust_norm
```

    Vector: [1, 2, 3, 4]
    L2 norm: 5.47723
    Expected: 5.47723


---

## 7. Performance Comparison

This section compares a simple CPU vector addition loop with a Thrust GPU implementation on 10 million elements.

The goal is not to prove that every GPU version is faster. The goal is to separate two different questions:

1. **How fast is the GPU computation once the data is already on the GPU?**
2. **How fast is the full workflow if data must be copied from CPU to GPU and back?**

Those are very different measurements. A single vector add is cheap: it performs one addition per element, so memory movement can dominate the total time. The example therefore reports both:

- **CPU time**: a regular C++ loop over `std::vector`.
- **GPU compute-only**: time for the Thrust `transform` after inputs are already in `device_vector`s.
- **GPU end-to-end**: time to copy CPU inputs to GPU, run the transform, and copy the result back.

The warm-up call before timing is intentional. The first CUDA operation can pay one-time setup costs such as context creation and allocation setup. Including that in the measured kernel time would make the GPU computation look misleadingly slow.

When reading the result, focus on the lesson: GPUs are strongest when you do enough work per transfer, reuse data already on the GPU, or chain many operations before copying results back to the CPU.



```python
%%writefile thrust_performance.cu
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

int main() {
    const int N = 10000000;  // 10 million elements

    // ---- CPU version ----
    std::vector<float> a_cpu(N, 1.0f);
    std::vector<float> b_cpu(N, 2.0f);
    std::vector<float> c_cpu(N);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < N; i++) {
        c_cpu[i] = a_cpu[i] + b_cpu[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_cpu - start_cpu).count() / 1000.0;

    // ---- GPU version with Thrust ----
    // Warm-up launch: the first CUDA call pays one-time context/allocation
    // setup. Timing it would unfairly penalize the GPU, so we run once first.
    thrust::device_vector<float> a_gpu(N, 1.0f);
    thrust::device_vector<float> b_gpu(N, 2.0f);
    thrust::device_vector<float> c_gpu(N);
    thrust::transform(a_gpu.begin(), a_gpu.end(), b_gpu.begin(),
                      c_gpu.begin(), thrust::plus<float>());
    cudaDeviceSynchronize();

    // (a) Compute-only: time just the kernel, data already on the GPU.
    auto start_k = std::chrono::high_resolution_clock::now();
    thrust::transform(a_gpu.begin(), a_gpu.end(), b_gpu.begin(),
                      c_gpu.begin(), thrust::plus<float>());
    cudaDeviceSynchronize();  // Wait for GPU to finish
    auto end_k = std::chrono::high_resolution_clock::now();
    double gpu_kernel_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                               end_k - start_k).count() / 1000.0;

    // (b) End-to-end: copy inputs host->device, compute, copy result back.
    auto start_e2e = std::chrono::high_resolution_clock::now();
    thrust::device_vector<float> a_d = a_cpu;   // host -> device
    thrust::device_vector<float> b_d = b_cpu;   // host -> device
    thrust::device_vector<float> c_d(N);
    thrust::transform(a_d.begin(), a_d.end(), b_d.begin(),
                      c_d.begin(), thrust::plus<float>());
    thrust::host_vector<float> c_h = c_d;       // device -> host
    cudaDeviceSynchronize();
    auto end_e2e = std::chrono::high_resolution_clock::now();
    double gpu_e2e_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_e2e - start_e2e).count() / 1000.0;

    // ---- Report ----
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Vector size: " << N << " elements\n\n";
    std::cout << "CPU time:           " << cpu_ms        << " ms\n";
    std::cout << "GPU compute-only:   " << gpu_kernel_ms << " ms";
    if (gpu_kernel_ms > 0) std::cout << "  -> " << cpu_ms / gpu_kernel_ms << "x speedup";
    std::cout << "\n";
    std::cout << "GPU end-to-end:     " << gpu_e2e_ms    << " ms  (includes host<->device copies)";
    if (gpu_e2e_ms > 0) std::cout << "  -> " << cpu_ms / gpu_e2e_ms << "x speedup";
    std::cout << "\n\n";
    std::cout << "Note: the kernel itself is ~100x faster, but for a single\n"
              << "cheap op the PCIe transfers dominate end-to-end time. This is\n"
              << "why minimizing host<->device transfers matters (see Section 1).\n";

    return 0;
}

```

    Overwriting thrust_performance.cu



```python
!nvcc -arch=sm_89 thrust_performance.cu -o thrust_performance
!./thrust_performance
```

    Vector size: 10000000 elements
    
    CPU time:           51.19 ms
    GPU compute-only:   0.52 ms  -> 97.87x speedup
    GPU end-to-end:     31.82 ms  (includes host<->device copies)  -> 1.61x speedup
    
    Note: the kernel itself is ~100x faster, but for a single
    cheap op the PCIe transfers dominate end-to-end time. This is
    why minimizing host<->device transfers matters (see Section 1).


---

## 8. Key Takeaways

### The Thrust Mental Model

A Thrust program is usually built from four pieces:

1. **Data containers**: `host_vector` for CPU memory and `device_vector` for GPU memory.
2. **Iterator ranges**: `begin()` and `end()` define what data an algorithm processes.
3. **Parallel algorithms**: `transform`, `reduce`, `sort`, `scan`, `copy`, and related primitives.
4. **Operations**: built-in function objects or custom functors that run for each element or pair of elements.

If the range points into a `device_vector`, Thrust can execute the algorithm on the GPU. If the range points into host data, the operation may execute on the CPU, depending on the execution policy and backend.

### When to Use Thrust

Good use cases:
- Sorting large datasets
- Reductions such as sum, min, max, and dot-product style patterns
- Prefix sums and scan operations
- Element-wise transformations
- Data reorganization and filtering
- Prototyping GPU algorithms before writing custom kernels

Less ideal use cases:
- Highly irregular algorithms with complex branching
- Algorithms that require carefully managed shared memory
- Cases where you need explicit control over blocks, warps, occupancy, or memory hierarchy
- Very small workloads where launch overhead and data transfers dominate

### Best Practices

1. **Minimize host-device transfers**: Move data to the GPU once, chain operations there, and copy back only final results.
2. **Prefer composition**: Many useful GPU workflows are combinations of `transform`, `reduce`, `scan`, `sort`, and `copy`.
3. **Use fancy iterators**: `constant_iterator`, `counting_iterator`, `transform_iterator`, and `zip_iterator` can remove temporary arrays.
4. **Make data movement visible**: Be clear when code copies from host to device or device to host.
5. **Profile real workloads**: Use Nsight Systems and Nsight Compute to understand transfer time, kernel time, and memory behavior.

### Performance Tips

- Thrust algorithms are optimized, but performance still depends on data size, memory bandwidth, and transfer cost.
- Transform iterators and zip iterators can reduce memory traffic by avoiding intermediate storage.
- Fusing operations can help. For example, `transform_reduce` can replace a separate transform followed by reduce.
- GPU acceleration usually pays off most when data stays on the GPU across multiple operations.

---

## 9. Further Resources

- **Official Documentation**: [NVIDIA Thrust Documentation](https://nvidia.github.io/cccl/thrust/)
- **CUDA C++ Programming Guide**: [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **GitHub Examples**: [Thrust Examples](https://github.com/NVIDIA/cccl/tree/main/thrust/examples)
- **Profiling Tools**: Nsight Systems and Nsight Compute are the modern NVIDIA tools for performance analysis.

---

## Practice Exercises

Try implementing these on your own:

1. **Variance calculation**: Compute the variance of a vector using `transform_iterator` and `reduce`.
2. **Dot product fusion**: Rewrite the zip iterator example using `thrust::transform_reduce`.
3. **Histogram**: Count occurrences of values in different bins.
4. **Matrix operations**: Implement matrix-vector multiplication.
5. **Filter operation**: Remove elements that do not satisfy a predicate.
6. **Performance experiment**: Vary vector size and compare CPU time, GPU compute-only time, and GPU end-to-end time.

A good next step is to take one CPU loop from your own C++ code and ask whether it can be expressed as a Thrust algorithm over one or more ranges.

