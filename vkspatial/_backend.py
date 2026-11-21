"""Zero-copy Vulkan bridge from libvkspatial.so to vknumpy's VkDevice and VkBuffers."""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_int64, c_size_t, c_uint32, c_void_p
from pathlib import Path
from typing import Optional

import numpy as _np
import vknumpy as vnp
from vknumpy._backend import lib as vkn_lib

_lib: Optional[ctypes.CDLL] = None
_initialized: bool = False


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    so_path = Path(__file__).resolve().parent / "_lib" / "libvkspatial.so"
    if not so_path.is_file():
        raise RuntimeError(f"libvkspatial.so not found at {so_path}. Run `pixi run build`.")
    lib = ctypes.CDLL(str(so_path))

    lib.vksp_init.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p]
    lib.vksp_init.restype = c_int

    lib.vksp_morton_2d.argtypes = [
        c_void_p, c_size_t,
        c_void_p, c_size_t,
        c_int64,
        c_float, c_float,
        c_float, c_float,
        c_void_p, c_size_t,
    ]
    lib.vksp_morton_2d.restype = c_int

    lib.vksp_point_in_polygon.argtypes = [
        c_void_p, c_size_t,
        c_void_p, c_size_t,
        c_int64,
        c_void_p, c_size_t,
        c_int,
        c_void_p, c_size_t,
    ]
    lib.vksp_point_in_polygon.restype = c_int

    lib.vksp_sjoin_polygons.argtypes = [
        c_void_p, c_size_t,
        c_void_p, c_size_t,
        c_int64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_int,
        c_void_p, c_size_t,
    ]
    lib.vksp_sjoin_polygons.restype = c_int

    _lib = lib
    return lib


def _ensure_init() -> ctypes.CDLL:
    global _initialized
    lib = _load_lib()
    if _initialized:
        return lib
    vl = vkn_lib()
    dev = vl.vkn_vk_get_device()
    qfam = vl.vkn_vk_get_compute_queue_family()
    if not dev or qfam == 0xFFFFFFFF:
        raise RuntimeError("vknumpy Vulkan device unavailable for vkspatial")
    submit_ptr = ctypes.cast(vl.vkn_vk_submit_compute, c_void_p).value
    sync_ptr = ctypes.cast(vl.vkn_vk_synchronize, c_void_p).value
    rc = lib.vksp_init(dev, qfam, submit_ptr, sync_ptr)
    if rc != 0:
        raise RuntimeError(f"vksp_init failed (rc={rc})")
    _initialized = True
    return lib


def morton_2d_gpu(
    x: vnp.ndarray,
    y: vnp.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> vnp.ndarray:
    """Compute 24-bit 2D Morton (Z-order) codes on GPU (zero-copy VkBuffer)."""
    lib = _ensure_init()
    vl = vkn_lib()
    xx = vnp.asarray(x, dtype=_np.float32).ravel().contiguous()
    yy = vnp.asarray(y, dtype=_np.float32).ravel().contiguous()
    if xx.size != yy.size:
        raise ValueError("x and y must have the same length")
    out = vnp.ndarray._empty_uninit((xx.size,), _np.int32)
    if xx.size == 0:
        return out
    rc = lib.vksp_morton_2d(
        vl.vkn_vk_get_raw_buffer(xx._buf.ptr), xx._offset,
        vl.vkn_vk_get_raw_buffer(yy._buf.ptr), yy._offset,
        xx.size,
        float(xmin), float(xmax),
        float(ymin), float(ymax),
        vl.vkn_vk_get_raw_buffer(out._buf.ptr), 0,
    )
    if rc != 0:
        raise RuntimeError(f"vksp_morton_2d failed (rc={rc})")
    return out


def point_in_polygon_gpu(x: vnp.ndarray, y: vnp.ndarray, poly_xy: vnp.ndarray) -> vnp.ndarray:
    """Test millions of (x, y) points against a polygon on GPU; returns int32 {0, 1} mask."""
    lib = _ensure_init()
    vl = vkn_lib()
    xx = vnp.asarray(x, dtype=_np.float32).ravel().contiguous()
    yy = vnp.asarray(y, dtype=_np.float32).ravel().contiguous()
    poly = vnp.asarray(poly_xy, dtype=_np.float32).ravel().contiguous()
    if xx.size != yy.size:
        raise ValueError("x and y must have the same length")
    if poly.size < 6 or (poly.size % 2) != 0:
        raise ValueError("poly_xy must have shape (M, 2) with M >= 3 vertices")
    n_verts = poly.size // 2
    out = vnp.ndarray._empty_uninit((xx.size,), _np.int32)
    if xx.size == 0:
        return out
    rc = lib.vksp_point_in_polygon(
        vl.vkn_vk_get_raw_buffer(xx._buf.ptr), xx._offset,
        vl.vkn_vk_get_raw_buffer(yy._buf.ptr), yy._offset,
        xx.size,
        vl.vkn_vk_get_raw_buffer(poly._buf.ptr), poly._offset,
        n_verts,
        vl.vkn_vk_get_raw_buffer(out._buf.ptr), 0,
    )
    if rc != 0:
        raise RuntimeError(f"vksp_point_in_polygon failed (rc={rc})")
    return out


def sjoin_polygons_gpu(
    x: vnp.ndarray,
    y: vnp.ndarray,
    verts_xy: vnp.ndarray,
    offsets: vnp.ndarray,
    bboxes: vnp.ndarray,
    n_polys: int,
) -> vnp.ndarray:
    """Single-pass zero-copy GPU multi-polygon spatial join; returns int32 zone_id (-1 if outside)."""
    lib = _ensure_init()
    vl = vkn_lib()
    xx = vnp.asarray(x, dtype=_np.float32).ravel().contiguous()
    yy = vnp.asarray(y, dtype=_np.float32).ravel().contiguous()
    vv = vnp.asarray(verts_xy, dtype=_np.float32).ravel().contiguous()
    oo = vnp.asarray(offsets, dtype=_np.int32).ravel().contiguous()
    bb = vnp.asarray(bboxes, dtype=_np.float32).ravel().contiguous()
    if xx.size != yy.size:
        raise ValueError("x and y must have the same length")
    out = vnp.ndarray._empty_uninit((xx.size,), _np.int32)
    if xx.size == 0 or n_polys <= 0:
        return out
    rc = lib.vksp_sjoin_polygons(
        vl.vkn_vk_get_raw_buffer(xx._buf.ptr), xx._offset,
        vl.vkn_vk_get_raw_buffer(yy._buf.ptr), yy._offset,
        xx.size,
        vl.vkn_vk_get_raw_buffer(vv._buf.ptr),
        vl.vkn_vk_get_raw_buffer(oo._buf.ptr),
        vl.vkn_vk_get_raw_buffer(bb._buf.ptr),
        int(n_polys),
        vl.vkn_vk_get_raw_buffer(out._buf.ptr), 0,
    )
    if rc != 0:
        raise RuntimeError(f"vksp_sjoin_polygons failed (rc={rc})")
    return out
