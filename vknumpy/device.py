"""Device inspection utilities for vknumpy."""

from __future__ import annotations

from ._backend import lib


def is_vulkan_available() -> bool:
    """Return True if vknumpy is executing on a Vulkan GPU via ggml-vulkan."""
    return bool(lib().vkn_is_vulkan())


def device_name() -> str:
    """Return the active ggml backend device name."""
    return lib().vkn_device_name().decode("utf-8", errors="replace")


def memory_info() -> dict[str, int]:
    """Return free and total device memory in bytes (0 on CPU fallback)."""
    l = lib()
    return {
        "free": int(l.vkn_device_free_memory()),
        "total": int(l.vkn_device_total_memory()),
    }


def device_info() -> dict[str, object]:
    """Return a summary dictionary of the active vknumpy backend device."""
    mem = memory_info()
    return {
        "backend": "ggml-vulkan" if is_vulkan_available() else "ggml-cpu",
        "vulkan": is_vulkan_available(),
        "name": device_name(),
        "free_memory_mb": round(mem["free"] / (1024 * 1024), 2),
        "total_memory_mb": round(mem["total"] / (1024 * 1024), 2),
    }
