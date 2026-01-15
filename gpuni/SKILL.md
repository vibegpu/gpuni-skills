---
name: gpuni
description: >-
  Cross-platform GPU kernel dialect — write once, run on NVIDIA (CUDA), AMD (HIP), and Intel/others (OpenCL 1.2).
  Use when: (1) writing portable GPU code across CUDA/HIP/OpenCL,
  (2) porting CUDA kernels to AMD or OpenCL,
  (3) creating/editing *.gu.cu files,
  (4) needing cross-vendor GPU compute,
  (5) using address-space qualifiers (__global/__local/__constant),
  (6) implementing portable atomics or shared memory,
  (7) debugging OpenCL render errors.
---

# gpuni

Write portable GPU kernels in CUDA-truth dialect. Compiles as CUDA/HIP, renders to OpenCL C 1.2.

## Critical Rules (always apply)

```cpp
// 0. Dialect file basics
// - Only include "gpuni.h" (no <...> system includes)

// 1. Entry signature
extern "C" __global__ void kernel_name(...)

// 2. All pointers AND aliases must have address space
__global float* output,           // param
__global float* p = output;       // alias - MUST keep __global
__local float* tile = smem + n;   // alias - MUST keep __local

// 3. Barriers must be uniform (all threads reach it)
__syncthreads();  // ✓ outside if
if (cond) __syncthreads();  // ✗ divergent = deadlock
```

## Workflow

1. Apply Critical Rules above for simple kernels
2. Need API details? Read `references/README.md`
3. Need code templates? Read `references/examples.md`
4. Compile error? Check `references/dialect.md` (especially for OpenCL)

## Host Pitfall

```cpp
// gu::Malloc returns device memory - NEVER dereference on host
float* d_x = gu::Malloc<float>(n);
d_x[0] = 1.0f;  // ✗ WRONG: segfault or undefined behavior
gu::Memcpy(d_x, h_x, bytes, gu::H2D);  // ✓ use Memcpy instead
```

## Review Checklist

- [ ] All pointers have `__global`/`__local`/`__constant`
- [ ] All pointer aliases retain address-space qualifier
- [ ] `__syncthreads()` reachable by all threads (no divergent barriers)
- [ ] Entry uses `extern "C" __global__ void`
- [ ] Dynamic smem param is last + `bindSharedMem()` called
- [ ] No warp intrinsics (`__shfl*`, `__ballot*`)
- [ ] No system includes in kernels (no `<...>`; only `"gpuni.h"` or `"gpuni/..."`)

## References

| File | When to read |
|------|--------------|
| `references/README.md` | Dialect rules + API (kernel & host) |
| `references/gpuni.h` | Implementation details, macro definitions |
| `references/examples.md` | Complete code templates |
| `references/dialect.md` | Compile errors (OpenCL error → fix mapping) |

## Package

| File | Purpose |
|------|---------|
| `references/gpuni.h` | Core header (backend detection, macros, portable APIs) |
| `references/render.c` | Offline CUDA→OpenCL renderer |
| `references/README.md` | Usage guide |
