"""GEMM benchmark comparing vknumpy (llama.cpp ggml-vulkan) vs CPU NumPy."""

import time
import numpy as np
import vknumpy as vnp


def bench_size(n: int, dtype=np.float32, warmup: int = 3, iters: int = 10) -> None:
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((n, n)).astype(dtype)
    b_np = rng.standard_normal((n, n)).astype(dtype)

    # CPU NumPy (skip slow scalar float16 CPU matmul for N > 512)
    if dtype == np.float16 and n > 512:
        cpu_ms = float("nan")
        cpu_gflops = float("nan")
        c_np = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.float16)
    else:
        for _ in range(warmup):
            _ = a_np @ b_np
        t0 = time.perf_counter()
        for _ in range(iters):
            c_np = a_np @ b_np
        cpu_ms = (time.perf_counter() - t0) * 1000.0 / iters
        cpu_gflops = (2.0 * n * n * n) / (cpu_ms * 1e-3) / 1e9

    # vknumpy (ggml-vulkan) — B pre-transposed vs standard B
    a_vk = vnp.array(a_np)
    b_vk = vnp.array(b_np)
    bt_vk = vnp.array(np.ascontiguousarray(b_np.T)).T  # zero-copy transposed view

    for _ in range(warmup):
        _ = a_vk @ b_vk
    t0 = time.perf_counter()
    for _ in range(iters):
        c_vk = a_vk @ b_vk
    vk_ms = (time.perf_counter() - t0) * 1000.0 / iters

    for _ in range(warmup):
        _ = a_vk @ bt_vk
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a_vk @ bt_vk
    vk_t_ms = (time.perf_counter() - t0) * 1000.0 / iters

    flops = 2.0 * n * n * n
    vk_gflops = flops / (vk_ms * 1e-3) / 1e9
    vk_t_gflops = flops / (vk_t_ms * 1e-3) / 1e9

    max_err = float(np.max(np.abs(c_vk.to_numpy().astype(np.float32) - c_np.astype(np.float32))))
    speedup_str = f"{cpu_ms / vk_t_ms:5.2f}x" if np.isfinite(cpu_ms) else "   n/a"
    cpu_str = f"{cpu_ms:7.2f} ms ({cpu_gflops:6.1f} GFLOPS)" if np.isfinite(cpu_ms) else "    (skipped: slow f16) "
    print(
        f"N={n:4d} ({np.dtype(dtype).name:7s}) | "
        f"NumPy CPU: {cpu_str} | "
        f"vknumpy: {vk_ms:7.2f} ms ({vk_gflops:6.1f} GFLOPS) | "
        f"vknumpy (B.T): {vk_t_ms:7.2f} ms ({vk_t_gflops:6.1f} GFLOPS) | "
        f"speedup: {speedup_str} | max_err={max_err:.2e}"
    )


def main() -> None:
    print("Device:", vnp.device_info())
    print("-" * 130)
    for n in (256, 512, 1024, 2048):
        bench_size(n, dtype=np.float32, warmup=2, iters=8)
    print("-" * 130)
    for n in (512, 1024, 2048):
        bench_size(n, dtype=np.float16, warmup=2, iters=8)


if __name__ == "__main__":
    main()
