"""Zero-copy Vulkan bridge from libvkshader.so to vknumpy's VkDevice and VkBuffers."""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_size_t, c_uint32, c_void_p
from pathlib import Path
from typing import Optional, Tuple

import numpy as _np
import vknumpy as vnp
from vknumpy._backend import lib as vkn_lib

_SHADE_HOW_MAP = {"eq_hist": 0, "log": 1, "cbrt": 2, "linear": 3}

_lib: Optional[ctypes.CDLL] = None
_initialized: bool = False
_scratch_stats: Optional[vnp.ndarray] = None
_scratch_cdf: Optional[vnp.ndarray] = None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    so_path = Path(__file__).resolve().parent / "_lib" / "libvkshader.so"
    if not so_path.is_file():
        raise RuntimeError(
            f"libvkshader.so not found at {so_path}. Build it with:\n"
            f"  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
        )
    lib = ctypes.CDLL(str(so_path))

    lib.vks_init.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p]
    lib.vks_init.restype = c_int

    lib.vks_shade_rgba.argtypes = [
        c_void_p, c_size_t,
        c_int, c_int,
        c_int, c_float,
        c_int,
        c_void_p, c_size_t,
        c_int, c_int,
        c_float, c_float, c_float, c_float,
        c_void_p,
        c_void_p,
        c_void_p, c_size_t,
    ]
    lib.vks_shade_rgba.restype = c_int

    _lib = lib
    return lib


def _ensure_init() -> ctypes.CDLL:
    global _initialized, _scratch_stats, _scratch_cdf
    lib = _load_lib()
    if _initialized:
        return lib

    vl = vkn_lib()
    dev = vl.vkn_vk_get_device()
    qfam = vl.vkn_vk_get_compute_queue_family()
    if not dev or qfam == 0xFFFFFFFF:
        raise RuntimeError("vknumpy Vulkan device is not available for zero-copy vkshader interop")

    # Pass C function pointers from libvknumpy.so directly into libvkshader.so
    submit_ptr = ctypes.cast(vl.vkn_vk_submit_compute, c_void_p).value
    sync_ptr = ctypes.cast(vl.vkn_vk_synchronize, c_void_p).value

    rc = lib.vks_init(dev, qfam, submit_ptr, sync_ptr)
    if rc != 0:
        raise RuntimeError(f"vks_init failed (rc={rc})")

    _scratch_stats = vnp.zeros((2050,), dtype=_np.int32)
    _scratch_cdf = vnp.zeros((2048,), dtype=_np.float32)
    _initialized = True
    return lib


def shade_rgba_gpu(
    grid: vnp.ndarray,
    how: str = "eq_hist",
    alpha: float = 235.0 / 255.0,
    bg: Optional[vnp.ndarray] = None,
    uv_rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    spread_px: int = 0,
) -> vnp.ndarray:
    """Shade a 2D int32 count grid with Fire colormap + optional GPU basemap compositing.

    Runs 100% on vknumpy's VkDevice with zero host intermediate copies.
    Returns a GPU int32 array of shape (H, W) whose bytes are packed RGBA8.
    """
    if how not in _SHADE_HOW_MAP:
        raise ValueError(f"Unsupported how={how!r}; expected one of {tuple(_SHADE_HOW_MAP)}")

    lib = _ensure_init()
    vl = vkn_lib()

    g = vnp.asarray(grid, dtype=_np.int32).contiguous()
    if g.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {g.shape}")
    height, width = int(g.shape[0]), int(g.shape[1])

    grid_vkbuf = vl.vkn_vk_get_raw_buffer(g._buf.ptr)
    bg_vkbuf = None
    bg_off = 0
    bg_w = 0
    bg_h = 0
    if bg is not None:
        bg_arr = vnp.asarray(bg, dtype=_np.int32).contiguous()
        if bg_arr.ndim != 2:
            raise ValueError(f"bg must be 2D packed RGBA int32, got shape {bg_arr.shape}")
        bg_h, bg_w = int(bg_arr.shape[0]), int(bg_arr.shape[1])
        bg_vkbuf = vl.vkn_vk_get_raw_buffer(bg_arr._buf.ptr)
        bg_off = bg_arr._offset

    assert _scratch_stats is not None and _scratch_cdf is not None
    stats_vkbuf = vl.vkn_vk_get_raw_buffer(_scratch_stats._buf.ptr)
    cdf_vkbuf = vl.vkn_vk_get_raw_buffer(_scratch_cdf._buf.ptr)

    out = vnp.ndarray._empty_uninit((height, width), _np.int32)
    out_vkbuf = vl.vkn_vk_get_raw_buffer(out._buf.ptr)

    rc = lib.vks_shade_rgba(
        grid_vkbuf, g._offset,
        width, height,
        _SHADE_HOW_MAP[how], float(alpha),
        int(spread_px),
        bg_vkbuf, bg_off,
        bg_w, bg_h,
        float(uv_rect[0]), float(uv_rect[1]), float(uv_rect[2]), float(uv_rect[3]),
        stats_vkbuf,
        cdf_vkbuf,
        out_vkbuf, 0,
    )
    if rc != 0:
        raise RuntimeError(f"vks_shade_rgba failed (rc={rc})")
    return out
