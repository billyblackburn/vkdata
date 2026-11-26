#!/usr/bin/env python3
"""Inspect your Vulkan GPU hardware specs & run a 1-second compute/bandwidth check.

Inspired by modular/mojo-gpu-puzzles' scripts/gpu_specs.py, tailored for any
Vulkan 1.3+ GPU (Steam Deck, AMD APU, Intel iGPU/Arc, older Radeon/NVIDIA, MoltenVK).

Usage:
    pixi run gpu-specs
    # or:
    PYTHONPATH=. python scripts/gpu_specs.py
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

# Ensure repo root is importable when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import vknumpy as vnp


def _parse_vulkaninfo() -> Dict[str, str]:
    """Query Vulkan compute limits from `vulkaninfo` if installed."""
    info: Dict[str, str] = {}
    if not shutil.which("vulkaninfo"):
        return info

    try:
        out = subprocess.check_output(
            ["vulkaninfo"], text=True, stderr=subprocess.DEVNULL, timeout=10
        )
    except Exception:
        return info

    patterns = {
        "api_version": r"apiVersion\s*=\s*([^\n]+)",
        "driver_name": r"driverName\s*=\s*([^\n]+)",
        "driver_info": r"driverInfo\s*=\s*([^\n]+)",
        "device_type": r"deviceType\s*=\s*([^\n]+)",
        "subgroup_size": r"subgroupSize\s*=\s*(\d+)",
        "shared_mem_bytes": r"maxComputeSharedMemorySize\s*=\s*(\d+)",
        "max_wg_invocations": r"maxComputeWorkGroupInvocations\s*=\s*(\d+)",
        "max_wg_size_x": r"maxComputeWorkGroupSize:\s*count\s*=\s*3\s*\n\s*(\d+)",
        "max_push_const": r"maxPushConstantsSize\s*=\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, out)
        if m:
            info[key] = m.group(1).strip()
    return info


@vnp.kernel(local_size=256)
def _bw_copy(a, out, n: int):
    i = global_id_x()
    if i < n:
        out[i] = a[i] + 1.0


def _quick_benchmarks() -> Dict[str, float]:
    """Run a ~1-second Vulkan GPU compute (GEMM) + memory bandwidth sanity check."""
    res: Dict[str, float] = {}

    # 1. FP32 1024x1024 GEMM (2 * N^3 FLOPs) via ggml-vulkan
    n = 1024
    rng = np.random.default_rng(0)
    a = vnp.array(rng.standard_normal((n, n)).astype(np.float32))
    b = vnp.array(rng.standard_normal((n, n)).astype(np.float32))
    _ = (a @ b).to_numpy()  # warmup

    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    _ = float(c[0, 0])  # sync
    dt = (time.perf_counter() - t0) / iters
    res["gemm_ms"] = dt * 1000.0
    res["gemm_gflops"] = (2.0 * n * n * n) / (dt * 1e9)

    # 2. 4M-element SPIR-V memory bandwidth test (read 4B + write 4B = 8B/elem)
    m = 4_000_000
    x = vnp.ones((m,), dtype=np.float32)
    y = vnp.zeros((m,), dtype=np.float32)
    grid = (m + 255) // 256
    _bw_copy[grid](x, y, m)
    _ = float(y[0])  # warmup

    bw_iters = 20
    t0 = time.perf_counter()
    for _ in range(bw_iters):
        _bw_copy[grid](x, y, m)
    _ = float(y[0])  # sync
    dt_bw = (time.perf_counter() - t0) / bw_iters
    res["bw_ms"] = dt_bw * 1000.0
    res["bw_gbs"] = (m * 8.0) / (dt_bw * 1e9)
    return res


def main() -> None:
    dev = vnp.device_info()
    vk = _parse_vulkaninfo()
    bench = _quick_benchmarks()

    total_gb = dev.get("total_memory_mb", 0.0) / 1024.0
    free_gb = dev.get("free_memory_mb", 0.0) / 1024.0
    shared_kb = int(vk["shared_mem_bytes"]) / 1024.0 if "shared_mem_bytes" in vk else 64.0
    subgroup = int(vk.get("subgroup_size", 32))

    line = "=" * 66
    print(f"\n{line}")
    print("  vkdata · Vulkan GPU Hardware & Performance Specs")
    print(line)
    print(f"  Device Name         : {dev.get('name', 'Unknown')}")
    print(f"  Backend             : {dev.get('backend', 'ggml-vulkan')} (Vulkan 1.3+)")
    if "device_type" in vk:
        print(f"  Device Type         : {vk['device_type']}")
    if "api_version" in vk:
        print(f"  Vulkan API Version  : {vk['api_version']}")
    if "driver_name" in vk:
        drv = vk["driver_name"] + (f" ({vk['driver_info']})" if "driver_info" in vk else "")
        print(f"  Driver              : {drv}")
    print(f"  Host OS / Arch      : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  GPU Memory (VRAM)   : {total_gb:.2f} GB total  |  {free_gb:.2f} GB free")
    print("-" * 66)
    print("  Compute & Workgroup Limits (used in Part II GPU Puzzles & vktorch)")
    print(f"  Wavefront / Subgroup: {subgroup} threads")
    print(f"  Shared Memory / WG  : {shared_kb:.0f} KB  (fast on-chip 'cutting board')")
    if "max_wg_invocations" in vk:
        print(f"  Max Threads / WG    : {vk['max_wg_invocations']}")
    if "max_push_const" in vk:
        print(f"  Max Push Constants  : {vk['max_push_const']} bytes")
    print("-" * 66)
    print("  Live 1-Second Vulkan GPU Sanity Benchmark")
    print(f"  FP32 GEMM (1024x1024): {bench['gemm_ms']:6.2f} ms  ->  {bench['gemm_gflops']:7.1f} GFLOPS")
    print(f"  SPIR-V Mem Bandwidth : {bench['bw_ms']:6.2f} ms  ->  {bench['bw_gbs']:7.1f} GB/s  (4M float32)")
    print("-" * 66)
    print("  Tips for Following Along in the Book:")
    print(f"  • Pick local_size multiples of {subgroup} (e.g. 64, 128, 256) in Part II GPU Puzzles.")
    print(f"  • Fused SPIR-V kernels (LayerNorm/RMSNorm/Softmax) use up to {shared_kb:.0f} KB shared mem.")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
