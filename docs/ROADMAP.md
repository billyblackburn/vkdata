# `vknumpy` Implementation Roadmap & Progress

`vknumpy` builds a NumPy drop-in on top of battle-tested open-source Vulkan compute backends (**`llama.cpp` / `ggml-vulkan`** for GEMM/BLAS and core tensor ops, **`Tencent/ncnn`** for extended reductions & math kernels, and **`VkFFT`** planned for FFTs) rather than maintaining a custom GLSL shader pack from scratch.

---

## Completed Phases

### Phase 0 — Core Foundation & `llama.cpp` `ggml-vulkan` GEMM/BLAS (Done)
- C/C++ bridge ([`cpp/vknumpy_ggml.h`](../cpp/vknumpy_ggml.h), [`cpp/vknumpy_ggml.cpp`](../cpp/vknumpy_ggml.cpp)) linked against `llama.cpp`'s `ggml` + `ggml-vulkan` (with `ggml-cpu` fallback).
- GPU-resident `ndarray` ([`vknumpy/ndarray.py`](../vknumpy/ndarray.py)) with zero-copy `.T` / `swapaxes(-1, -2)` tracking, `__array__`, and `__array_ufunc__`.
- 2D–4D batched GEMM (`matmul` / `@` with zero-copy `B.T` fast path, `float32` & `float16`), `dot`, `vdot`, `inner`, `outer`, `tensordot`, 2-operand `einsum` GPU lowering, `linalg.norm`, `linalg.multi_dot`, `linalg.matrix_power`.
- Core elementwise ufuncs (`+`, `-`, `*`, `/` with broadcasting and fused `scale_bias`, `abs`, `negative`, `square`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `tanh`, `clip`, `relu`, `sigmoid`, `silu`, `gelu`) and reductions (`sum`, `mean`, `var`, `std`, `argmax`, `argmin`).
- Precision control: defaults to full IEEE FP32 (`GGML_VK_DISABLE_F16=1`), with opt-in `VKNUMPY_FAST_FP16=1` for ~1.0–1.2 TFLOPS FP16-dot2 GEMM.

### Phase 1 — Native `ggml-vulkan` Creation, Unary, Manipulation, Sorting & Triangular Solve (Done — commit `ebd5a00`)
Wired up **13 additional native `ggml-vulkan` C ABI kernels** and **7 unary ops** already compiled into `llama.cpp` (zero new shaders):
- **GPU-native creation (zero host upload)**: `zeros`, `ones`, `full` (`ggml_fill`), `arange` (`ggml_arange`), `diag` (`ggml_diag`), `tril`, `triu` (`ggml_tri`).
- **Unary math**: `floor`, `ceil`, `round` / `rint`, `trunc`, `sign`, `expm1`, `log1p`, `sinh`, `cosh`, `reciprocal`, `cbrt`, `heaviside`.
- **Manipulation & gather**: `concatenate`, `stack`, `vstack`, `hstack`, `dstack` (`ggml_concat`), `pad` (`ggml_pad_ext`), `roll` (`ggml_roll`), `tile` & `broadcast_to` (`ggml_repeat`), `take` (`ggml_get_rows`), `split`, `array_split`, `hsplit`, `vsplit`, `flip`.
- **Sorting & scans**: `cumsum` (`ggml_cumsum`), `argsort` (`ggml_argsort`), `sort`, `top_k` (`ggml_top_k`).
- **Linear algebra**: `linalg.solve_triangular` (`ggml_solve_tri`).

### Phase 2 — GPU Comparisons, `where`, Logicals & `Tencent/ncnn` Vulkan Math/Reductions (Done — commit `32d9a0e`)
Combined fused `ggml-vulkan` compute graphs with **[Tencent `ncnn`](https://github.com/Tencent/ncnn)** (BSD-3-Clause) Vulkan kernels in [`cpp/vknumpy_ncnn.cpp`](../cpp/vknumpy_ncnn.cpp) — zero hand-written GLSL shaders:
- **Fused `ggml-vulkan` ops**:
  - `maximum`, `minimum`, and all 6 comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=` / `greater`, `less`, `greater_equal`, `less_equal`, `equal`, `not_equal`) with 4D broadcasting.
  - Fused GPU ternary `where(cond, x, y)`.
  - `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `all`, `any`, `count_nonzero`, `isclose`, `array_equal`, and floored `remainder` (`%`).
- **`Tencent/ncnn` Vulkan GPU kernels** (gracefully optional if `third_party/ncnn` is absent):
  - **Reductions**: `max`, `min`, `prod` across full arrays and arbitrary axes (`ncnn::Reduction`).
  - **Binary math**: `power(a, b)`, `arctan2`, `fmod`, `floor_divide` (`//`), `logaddexp` (`ncnn::BinaryOp`).
  - **Inverse trig & hyperbolic**: `arcsin`, `arccos`, `arctan`, `arcsinh` (`ncnn::UnaryOp`), plus `arccosh`, `arctanh`.

---

## Upcoming Phases

### Phase 3 — Full `ndarray` Semantics & NumPy Protocols
- General strided views `(shape, strides, offset)` for arbitrary slicing (`a[:, 1:10:2]`, `a[::-1]`, `newaxis`).
- In-place operators (`+=`, `-=`, `*=`, `/=`, `**=`) using `ggml_*_inplace`.
- Boolean masking (`a[mask]`, `a[mask] = val`).
- `__array_function__` protocol for transparent NumPy ecosystem interop.
- High-level helpers: `meshgrid`, `logspace`, `geomspace`, `diff`, `gradient`, `unique`, `nonzero`, `searchsorted`, `percentile`, `quantile`, `median`, `histogram`, `corrcoef`, `cov`, `nan*` reductions, `kron`, `cross`.

### Phase 4 — Extended `vknumpy.linalg`, `vknumpy.fft` (`VkFFT`), & `vknumpy.random`
- GPU blocked Cholesky & LU factorizations built on `ggml_mul_mat` + `ggml_solve_tri` to accelerate `linalg.solve`, `inv`, `det`, `cholesky`.
- `vknumpy.fft` backed by header-only **[`VkFFT`](https://github.com/DTolm/VkFFT)** (MIT) for GPU `fft`, `ifft`, `rfft`, `irfft`, `fft2`.
- Expanded `vknumpy.random` distributions (`choice`, `permutation`, `binomial`, `poisson`, `exponential`, `gamma`, `beta`).

### Phase 5 — Lazy Graph Fusion, Discrete GPU Polish & `llama.cpp` Extensions
- Lazy graph recording (`with vknumpy.fuse():`) to compile chained ops into a single `ggml_cgraph` submission.
- `bfloat16` (`GGML_TYPE_BF16`) and opt-in quantized `llama.cpp` matmul dtypes (`Q8_0`, `Q4_0`, `Q4_K`).
- **Multi-GPU & discrete GPU optimizations**:
  - Explicit device selection API (`vknumpy.list_devices()`, `vknumpy.set_device(idx)`) across both `ggml-vulkan` and `ncnn` (currently defaults to device `0`; `GGML_VK_VISIBLE_DEVICES` can steer `ggml-vulkan`).
  - Zero-copy VRAM sharing between `ggml-vulkan` and `ncnn` (via `VK_KHR_external_memory` or shared `VkDevice`/buffer import) to eliminate the PCIe host-staging copy on `ncnn`-backed ops for discrete GPUs.

---

## Hardware Support (APUs & Discrete GPUs)

`vknumpy` targets standard Vulkan 1.3+ compute and works on **both APUs/iGPUs and discrete GPUs** (AMD Radeon, NVIDIA GeForce/RTX, Intel Arc, and macOS via MoltenVK):
- **All `ggml-vulkan` ops stay 100% in VRAM** with zero PCIe round-trips between chained ops, and automatically use hardware matrix cores (`GL_KHR_cooperative_matrix` / Tensor Cores / WMMA) when supported by the GPU driver.
- **`ncnn`-backed ops** (`max`, `min`, `prod`, `power`, `arctan2`, inverse trig, `fmod`, `floor_divide`, `logaddexp`) currently stage through host memory between `ggml-vulkan` and `ncnn`'s `VkDevice` — negligible on UMA APUs and functional on discrete GPUs, with zero-copy VRAM sharing planned in Phase 5.

### Phase 6 — API Coverage Matrix, Differential Tests & Packaging
- Automated NumPy API coverage tracker.
- Per-module differential test suite (`tests/test_*.py`).
- `scikit-build-core` packaging for `pip install .`.
