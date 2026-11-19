# vknumpy

**`vknumpy`** is a GPU-accelerated **NumPy drop-in** powered by [`llama.cpp`](https://github.com/ggml-org/llama.cpp)'s **`ggml-vulkan`** GEMM and BLAS compute shaders.

It lets you run matrix multiplications, batched GEMM, `einsum` contractions, elementwise ufuncs, and reductions directly on **any Vulkan 1.3+ GPU** (AMD RADV, NVIDIA, Intel, Apple/MoltenVK) using familiar NumPy syntax — no CUDA or ROCm required.

> **Status / Limitations:** This project is an early prototype — **most of the NumPy API is not implemented yet.** GPU acceleration currently covers `float32`/`float16` GEMM & BLAS (`matmul`, `dot`, `outer`, 2-operand `einsum`), core elementwise ufuncs/broadcasting, and common reductions (`sum`, `mean`, `var`, `std`, `argmax`). Unsupported dtypes (such as `float64`/`int64`) and some higher-level routines (such as `linalg.solve`, `svd`, `qr`, `concatenate`) currently fall back to CPU NumPy under the hood.

## What It Does

- **Direct `ggml-vulkan` C/C++ bridge (`libvknumpy.so`)**: Allocates GPU-resident buffers (`VknBuffer`) and dispatches compute graphs (`ggml_cgraph` + persistent `ggml_gallocr`) directly to `llama.cpp`'s Vulkan shaders, with automatic `ggml-cpu` fallback if no Vulkan GPU is available.
- **Zero-copy transposed views**: `B.T` and `swapaxes(-1, -2)` track transposed layout lazily, so `A @ B.T` passes straight into `ggml_mul_mat` without an extra transpose copy.
- **GEMM & BLAS**: `matmul` / `@` (2D, 3D, 4D batched + batch broadcasting, `float32` & `float16`), `dot`, `vdot`, `inner`, `outer`, `tensordot`, `einsum` (automatic GPU GEMM lowering for 2-operand contractions), and `linalg` (`norm`, `multi_dot`, `matrix_power`, `solve`, `inv`, `svd`, `qr`, `eigh`, etc.).
- **Elementwise ufuncs & reductions**: `+`, `-`, `*`, `/` with broadcasting and fused scalar `scale_bias`, `abs`, `negative`, `square`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `tanh`, `clip`, ML activations (`relu`, `sigmoid`, `silu`, `gelu`), and `sum`, `mean`, `var`, `std`, `argmax`, `argmin`.
- **Precision control**:
  - Defaults to **full IEEE FP32 precision** (`GGML_VK_DISABLE_F16=1`) for NumPy drop-in parity.
  - Set `VKNUMPY_FAST_FP16=1` to enable `ggml-vulkan`'s FP16-dot2 GEMM path (~1.0–1.2 TFLOPS on AMD RADV VanGogh / Steam Deck).

## Installation

### Prerequisites
- A Vulkan-capable GPU + driver (`libvulkan.so`) and `glslc` (from `shaderc` or the [Vulkan SDK](https://vulkan.lunarg.com/) >= 1.4).
- [`pixi`](https://pixi.sh/) (recommended — installs Python, NumPy, CMake, Ninja, GCC, and SPIRV-Headers automatically), **or** a system C++17 compiler, CMake >= 3.20, Ninja, Vulkan 1.4+ headers, SPIRV-Headers, and Python >= 3.10 with NumPy.

### Using `pixi` (Recommended)

```bash
git clone <your-repo-url> vknumpy
cd vknumpy

# 1. Install Python & C++ build toolchain into the workspace
pixi install

# 2. Fetch llama.cpp (shallow clone)
pixi run setup

# 3. Build ggml-vulkan + libvknumpy.so
#    If your system Vulkan headers are < 1.4, point VULKAN_SDK to your Vulkan 1.4+ SDK:
#    export VULKAN_SDK=/path/to/vulkansdk/1.4.x/x86_64
pixi run build

# 4. Install vknumpy in editable mode (optional, or use PYTHONPATH=.)
pixi run pip install --no-build-isolation --no-deps -e .

# 5. Run tests & GEMM benchmark
pixi run test
pixi run bench
```

### Manual Build (without `pixi`)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
pip install --no-build-isolation --no-deps -e .
```

## Short Example

```python
import vknumpy as np

np.random.seed(0)
print("Device:", np.device_info())

# Allocate GPU-resident float32 arrays
x = np.random.randn(1024, 1024)
w = np.random.randn(1024, 1024)

# GEMM on Vulkan GPU via llama.cpp (w.T uses zero-copy transposed fast path)
y = x @ w.T

# Chained elementwise ufuncs & reductions stay on GPU
z = np.relu(y * 0.5 + 0.1)
row_means = z.mean(axis=-1)

# Batched attention-style contraction lowered directly to GPU GEMM
q = np.random.randn(2, 4, 64, 64)
k = np.random.randn(2, 4, 64, 64)
scores = np.einsum("bhqd,bhkd->bhqk", q, k)

# Download back to standard NumPy when needed
host_arr = row_means.to_numpy()
print("Output shape:", host_arr.shape, "mean:", round(float(row_means.mean()), 4), "scores shape:", scores.shape)
```

Output:

```text
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Custom GPU 0405 (RADV VANGOGH) (radv) | uma: 1 | rebar: 1 | fp16: 0 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
Device: {'backend': 'ggml-vulkan', 'vulkan': True, 'name': 'AMD Custom GPU 0405 (RADV VANGOGH)', 'free_memory_mb': 11524.56, 'total_memory_mb': 12288.0}
Output shape: (1024,) mean: 6.433 scores shape: (2, 4, 64, 64)
```

## Benchmarks (AMD Custom GPU 0405 / RADV VanGogh)

| Size / Dtype | Mode | NumPy CPU | `vknumpy` (`A @ B`) | `vknumpy` (`A @ B.T`) | Speedup (`B.T`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `1024×1024` `float32` | IEEE FP32 (default) | 9.34 ms (230 GFLOPS) | 4.48 ms (479 GFLOPS) | 4.06 ms (529 GFLOPS) | **2.30x** |
| `2048×2048` `float32` | IEEE FP32 (default) | 67.12 ms (256 GFLOPS) | 29.18 ms (589 GFLOPS) | 22.77 ms (755 GFLOPS) | **2.95x** |
| `2048×2048` `float32` | `VKNUMPY_FAST_FP16=1` | 70.75 ms (243 GFLOPS) | 18.80 ms (914 GFLOPS) | 17.10 ms (**1.00 TFLOPS**) | **4.14x** |
| `2048×2048` `float16` | `VKNUMPY_FAST_FP16=1` | *(slow scalar CPU)* | 15.61 ms (1.10 TFLOPS) | 14.52 ms (**1.18 TFLOPS**) | — |

## Project Layout

- [`cpp/vknumpy_ggml.h`](cpp/vknumpy_ggml.h) / [`cpp/vknumpy_ggml.cpp`](cpp/vknumpy_ggml.cpp): C ABI wrapping `ggml-vulkan` buffers and compute graphs.
- [`vknumpy/_backend.py`](vknumpy/_backend.py): `ctypes` loader and RAII `DeviceBuffer`.
- [`vknumpy/ndarray.py`](vknumpy/ndarray.py): GPU-resident `ndarray` with NumPy operators and protocols.
- [`vknumpy/ops.py`](vknumpy/ops.py): GEMM/BLAS, `einsum` lowering, ufuncs, reductions, and shape ops.
- [`vknumpy/linalg.py`](vknumpy/linalg.py): Linear algebra routines.
- [`vknumpy/creation.py`](vknumpy/creation.py): Array constructors and `random` submodule.
- [`tests/test_vknumpy.py`](tests/test_vknumpy.py) & [`benchmarks/bench_gemm.py`](benchmarks/bench_gemm.py): Verification suite and GEMM benchmark.
