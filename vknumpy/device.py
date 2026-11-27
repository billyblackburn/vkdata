"""Device inspection and RMM/PyTorch-style Vulkan VRAM profiling utilities for vknumpy."""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import Iterator

from ._backend import _mem_tracker, lib

_MB = 1024 * 1024


def is_vulkan_available() -> bool:
    """Return True if vknumpy is executing on a Vulkan GPU via ggml-vulkan."""
    return bool(lib().vkn_is_vulkan())


def device_name() -> str:
    """Return the active ggml backend device name."""
    return lib().vkn_device_name().decode("utf-8", errors="replace")


def memory_info() -> dict[str, int | float]:
    """Return device VRAM and live Vulkan buffer allocation diagnostics."""
    l = lib()
    free = int(l.vkn_device_free_memory())
    total = int(l.vkn_device_total_memory())
    snap = _mem_tracker.snapshot()
    return {
        "free": free,
        "total": total,
        "allocated": snap["allocated"],
        "peak": snap["peak"],
        "live_buffers": snap["live_buffers"],
        "total_allocs": snap["total_allocs"],
        "total_frees": snap["total_frees"],
        "upload_bytes": snap["upload_bytes"],
        "download_bytes": snap["download_bytes"],
        "allocated_mb": round(snap["allocated"] / _MB, 3),
        "peak_mb": round(snap["peak"] / _MB, 3),
        "free_mb": round(free / _MB, 2),
        "total_mb": round(total / _MB, 2),
    }


def memory_allocated() -> int:
    """Return live Vulkan VRAM bytes currently held by vknumpy buffers."""
    return int(_mem_tracker.snapshot()["allocated"])


def max_memory_allocated() -> int:
    """Return peak Vulkan VRAM bytes allocated since startup or last reset."""
    return int(_mem_tracker.snapshot()["peak"])


def reset_peak_memory_stats() -> None:
    """Reset the peak Vulkan VRAM high-water mark to current live usage."""
    _mem_tracker.reset_peak()


def empty_cache() -> int:
    """Force garbage collection of unreferenced Vulkan buffers and return bytes freed."""
    before = memory_allocated()
    gc.collect()
    after = memory_allocated()
    return max(0, before - after)


def memory_summary() -> str:
    """Return a formatted Vulkan VRAM summary table (like torch.cuda.memory_summary)."""
    m = memory_info()
    lines = [
        f"=== Vulkan VRAM Summary ({device_name()}) ===",
        f"  Live Allocated : {m['allocated_mb']:8.3f} MB ({m['live_buffers']:,} buffers)",
        f"  Peak Allocated : {m['peak_mb']:8.3f} MB",
        f"  Device Free    : {m['free_mb']:8.2f} MB / {m['total_mb']:.2f} MB",
        f"  Alloc / Free   : {m['total_allocs']:,} / {m['total_frees']:,}",
        f"  H2D / D2H Copy : {m['upload_bytes'] / _MB:8.3f} MB up / {m['download_bytes'] / _MB:8.3f} MB down",
    ]
    return "\n".join(lines)


@contextmanager
def profile_memory(label: str = "", *, verbose: bool = False) -> Iterator[dict[str, float | int]]:
    """Context manager measuring Vulkan VRAM delta, peak, allocations, and H2D/D2H transfers."""
    start = _mem_tracker.snapshot()
    prev_peak = start["peak"]
    _mem_tracker.reset_peak()
    t0 = time.perf_counter()
    report: dict[str, float | int] = {}
    try:
        yield report
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        end = _mem_tracker.snapshot()
        # Preserve global peak if earlier peak was higher
        with _mem_tracker._lock:
            if prev_peak > _mem_tracker.peak_bytes:
                _mem_tracker.peak_bytes = prev_peak
        report.update(
            {
                "label": label,
                "elapsed_ms": round(elapsed_ms, 3),
                "delta_bytes": end["allocated"] - start["allocated"],
                "delta_mb": round((end["allocated"] - start["allocated"]) / _MB, 3),
                "peak_bytes": end["peak"],
                "peak_mb": round(end["peak"] / _MB, 3),
                "allocs": end["total_allocs"] - start["total_allocs"],
                "frees": end["total_frees"] - start["total_frees"],
                "upload_mb": round((end["upload_bytes"] - start["upload_bytes"]) / _MB, 3),
                "download_mb": round((end["download_bytes"] - start["download_bytes"]) / _MB, 3),
            }
        )
        if verbose:
            tag = f"[{label}] " if label else ""
            print(
                f"{tag}VRAM delta={report['delta_mb']:+.3f} MB | peak={report['peak_mb']:.3f} MB | "
                f"allocs={report['allocs']} | H2D={report['upload_mb']:.3f} MB | "
                f"D2H={report['download_mb']:.3f} MB | {report['elapsed_ms']:.2f} ms"
            )


def device_info() -> dict[str, object]:
    """Return a summary dictionary of the active vknumpy backend device."""
    mem = memory_info()
    return {
        "backend": "ggml-vulkan" if is_vulkan_available() else "ggml-cpu",
        "vulkan": is_vulkan_available(),
        "name": device_name(),
        "free_memory_mb": mem["free_mb"],
        "total_memory_mb": mem["total_mb"],
        "allocated_memory_mb": mem["allocated_mb"],
        "peak_memory_mb": mem["peak_mb"],
    }
