"""Low-level ctypes bridge to libvknumpy.so (llama.cpp ggml-vulkan backend)."""

from __future__ import annotations

import atexit
import ctypes
import os
from pathlib import Path
from typing import Optional
import numpy as _np

# Dtypes matching cpp/vknumpy_ggml.h
VKN_DTYPE_F32 = 0
VKN_DTYPE_F16 = 1
VKN_DTYPE_I32 = 2

# Binary ops
VKN_BIN_ADD = 0
VKN_BIN_SUB = 1
VKN_BIN_MUL = 2
VKN_BIN_DIV = 3

# Unary ops
VKN_UNARY_NEG     = 0
VKN_UNARY_ABS     = 1
VKN_UNARY_SQR     = 2
VKN_UNARY_SQRT    = 3
VKN_UNARY_EXP     = 4
VKN_UNARY_LOG     = 5
VKN_UNARY_SIN     = 6
VKN_UNARY_COS     = 7
VKN_UNARY_TANH    = 8
VKN_UNARY_RELU    = 9
VKN_UNARY_SIGMOID = 10
VKN_UNARY_SILU    = 11
VKN_UNARY_GELU    = 12
VKN_UNARY_FLOOR   = 13
VKN_UNARY_CEIL    = 14
VKN_UNARY_ROUND   = 15
VKN_UNARY_TRUNC   = 16
VKN_UNARY_SGN     = 17
VKN_UNARY_EXPM1   = 18
VKN_UNARY_STEP    = 19

# Triangular mask types
VKN_TRI_UPPER_DIAG = 0
VKN_TRI_UPPER      = 1
VKN_TRI_LOWER_DIAG = 2
VKN_TRI_LOWER      = 3

NP_TO_VKN_DTYPE = {
    _np.dtype(_np.float32): VKN_DTYPE_F32,
    _np.dtype(_np.float16): VKN_DTYPE_F16,
    _np.dtype(_np.int32):   VKN_DTYPE_I32,
}

VKN_TO_NP_DTYPE = {
    VKN_DTYPE_F32: _np.dtype(_np.float32),
    VKN_DTYPE_F16: _np.dtype(_np.float16),
    VKN_DTYPE_I32: _np.dtype(_np.int32),
}

NATIVE_FLOAT_DTYPES = (_np.dtype(_np.float32), _np.dtype(_np.float16))
NATIVE_DTYPES = (_np.dtype(_np.float32), _np.dtype(_np.float16), _np.dtype(_np.int32))


def _find_library() -> Path:
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "_lib" / "libvknumpy.so",
        pkg_dir.parent / "build" / "libvknumpy.so",
        pkg_dir / "_lib" / "libvknumpy.dylib",
        pkg_dir / "_lib" / "vknumpy.dll",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError(
        "Could not find libvknumpy shared library. "
        "Build it first with `cmake -B build -G Ninja && cmake --build build`."
    )


class _LibHandle:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._initialized = False

    def get(self) -> ctypes.CDLL:
        if self._lib is None:
            # Default to true IEEE FP32 GEMM precision for NumPy drop-in parity
            # unless the user opts into FP16-dot2 acceleration via VKNUMPY_FAST_FP16=1
            if "GGML_VK_DISABLE_F16" not in os.environ:
                if os.environ.get("VKNUMPY_FAST_FP16", "0") not in ("1", "true", "TRUE"):
                    os.environ["GGML_VK_DISABLE_F16"] = "1"
            lib_path = _find_library()
            lib = ctypes.CDLL(str(lib_path))
            self._bind(lib)
            rc = lib.vkn_init()
            if rc != 0:
                err = lib.vkn_last_error().decode("utf-8", errors="replace")
                raise RuntimeError(f"vkn_init() failed ({rc}): {err}")
            self._lib = lib
            self._initialized = True
            atexit.register(self._shutdown)
        return self._lib

    def _shutdown(self) -> None:
        if self._lib is not None and self._initialized:
            self._initialized = False
            try:
                self._lib.vkn_shutdown()
            except Exception:
                pass

    @staticmethod
    def _bind(lib: ctypes.CDLL) -> None:
        c_void_p = ctypes.c_void_p
        c_size_t = ctypes.c_size_t
        c_int = ctypes.c_int
        c_int64_p = ctypes.POINTER(ctypes.c_int64)
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float = ctypes.c_float
        c_char_p = ctypes.c_char_p

        lib.vkn_init.argtypes = []
        lib.vkn_init.restype = c_int

        lib.vkn_shutdown.argtypes = []
        lib.vkn_shutdown.restype = None

        lib.vkn_is_vulkan.argtypes = []
        lib.vkn_is_vulkan.restype = c_int

        lib.vkn_device_name.argtypes = []
        lib.vkn_device_name.restype = c_char_p

        lib.vkn_device_free_memory.argtypes = []
        lib.vkn_device_free_memory.restype = c_size_t

        lib.vkn_device_total_memory.argtypes = []
        lib.vkn_device_total_memory.restype = c_size_t

        lib.vkn_last_error.argtypes = []
        lib.vkn_last_error.restype = c_char_p

        lib.vkn_buf_alloc.argtypes = [c_size_t]
        lib.vkn_buf_alloc.restype = c_void_p

        lib.vkn_buf_free.argtypes = [c_void_p]
        lib.vkn_buf_free.restype = None

        lib.vkn_buf_size.argtypes = [c_void_p]
        lib.vkn_buf_size.restype = c_size_t

        lib.vkn_buf_upload.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t]
        lib.vkn_buf_upload.restype = c_int

        lib.vkn_buf_download.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t]
        lib.vkn_buf_download.restype = c_int

        lib.vkn_buf_copy.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t, c_size_t]
        lib.vkn_buf_copy.restype = c_int

        lib.vkn_matmul.argtypes = [
            c_void_p, c_size_t, c_int64_p, c_int,
            c_void_p, c_size_t, c_int64_p, c_int,
            c_int,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_matmul.restype = c_int

        lib.vkn_dot_1d.argtypes = [
            c_void_p, c_size_t,
            c_void_p, c_size_t,
            ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_dot_1d.restype = c_int

        lib.vkn_outer.argtypes = [
            c_void_p, c_size_t, ctypes.c_int64,
            c_void_p, c_size_t, ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_outer.restype = c_int

        lib.vkn_binary_op.argtypes = [
            c_int,
            c_void_p, c_size_t, c_int64_p,
            c_void_p, c_size_t, c_int64_p,
            c_void_p, c_size_t, c_int64_p,
            c_int,
            c_int,
        ]
        lib.vkn_binary_op.restype = c_int

        lib.vkn_unary_op.argtypes = [
            c_int,
            c_void_p, c_size_t,
            ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_unary_op.restype = c_int

        lib.vkn_scale_bias.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64,
            c_float, c_float,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_scale_bias.restype = c_int

        lib.vkn_clamp.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64,
            c_float, c_float,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_clamp.restype = c_int

        lib.vkn_sum_all.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_sum_all.restype = c_int

        lib.vkn_sum_rows.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64, ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_sum_rows.restype = c_int

        lib.vkn_mean_rows.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64, ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_mean_rows.restype = c_int

        lib.vkn_argmax_rows.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64, ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_argmax_rows.restype = c_int

        lib.vkn_transpose_cont.argtypes = [
            c_void_p, c_size_t,
            c_int64_p, c_int_p, c_int,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_transpose_cont.restype = c_int

        lib.vkn_cast.argtypes = [
            c_void_p, c_size_t, c_int,
            ctypes.c_int64,
            c_void_p, c_size_t, c_int,
        ]
        lib.vkn_cast.restype = c_int

        # Phase 1 bindings
        lib.vkn_fill.argtypes = [c_void_p, c_size_t, ctypes.c_int64, c_float, c_int]
        lib.vkn_fill.restype = c_int

        lib.vkn_arange.argtypes = [c_float, c_float, c_float, c_void_p, c_size_t, c_int]
        lib.vkn_arange.restype = c_int

        lib.vkn_diag.argtypes = [c_void_p, c_size_t, ctypes.c_int64, c_void_p, c_size_t, c_int]
        lib.vkn_diag.restype = c_int

        lib.vkn_tri.argtypes = [
            c_void_p, c_size_t,
            c_int64_p, c_int,
            c_int,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_tri.restype = c_int

        lib.vkn_concat.argtypes = [
            c_void_p, c_size_t, c_int64_p,
            c_void_p, c_size_t, c_int64_p,
            c_int, c_int,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_concat.restype = c_int

        lib.vkn_pad.argtypes = [
            c_void_p, c_size_t,
            c_int64_p, c_int,
            c_int_p, c_int_p,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_pad.restype = c_int

        lib.vkn_roll.argtypes = [
            c_void_p, c_size_t,
            c_int64_p, c_int,
            c_int_p,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_roll.restype = c_int

        lib.vkn_repeat.argtypes = [
            c_void_p, c_size_t, c_int64_p,
            c_void_p, c_size_t, c_int64_p,
            c_int,
            c_int,
        ]
        lib.vkn_repeat.restype = c_int

        lib.vkn_cumsum_rows.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64, ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_cumsum_rows.restype = c_int

        lib.vkn_argsort_rows.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64, ctypes.c_int64,
            c_int,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_argsort_rows.restype = c_int

        lib.vkn_top_k_rows.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64, ctypes.c_int64,
            c_int,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_top_k_rows.restype = c_int

        lib.vkn_get_rows.argtypes = [
            c_void_p, c_size_t,
            ctypes.c_int64, ctypes.c_int64,
            c_void_p, c_size_t,
            ctypes.c_int64,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_get_rows.restype = c_int

        lib.vkn_solve_tri.argtypes = [
            c_void_p, c_size_t, c_int64_p,
            c_void_p, c_size_t, c_int64_p,
            c_int,
            c_int, c_int, c_int,
            c_void_p, c_size_t,
            c_int,
        ]
        lib.vkn_solve_tri.restype = c_int


_handle = _LibHandle()


def lib() -> ctypes.CDLL:
    return _handle.get()


def check_rc(rc: int, op_name: str) -> None:
    if rc != 0:
        err = lib().vkn_last_error().decode("utf-8", errors="replace")
        raise RuntimeError(f"{op_name} failed (rc={rc}): {err}")


class DeviceBuffer:
    """Owns a ggml-vulkan device allocation and frees it on GC."""

    __slots__ = ("ptr", "nbytes", "_lib")

    def __init__(self, nbytes: int) -> None:
        self._lib = lib()
        self.nbytes = int(nbytes)
        self.ptr = self._lib.vkn_buf_alloc(self.nbytes)
        if not self.ptr:
            err = self._lib.vkn_last_error().decode("utf-8", errors="replace")
            raise MemoryError(f"Failed to allocate {nbytes} bytes on Vulkan device: {err}")

    def upload(self, arr: _np.ndarray, offset: int = 0) -> None:
        c_arr = _np.ascontiguousarray(arr)
        if c_arr.nbytes == 0:
            return
        rc = self._lib.vkn_buf_upload(
            self.ptr, offset, c_arr.ctypes.data_as(ctypes.c_void_p), c_arr.nbytes
        )
        check_rc(rc, "vkn_buf_upload")

    def download(self, shape: tuple[int, ...], dtype: _np.dtype, offset: int = 0) -> _np.ndarray:
        out = _np.empty(shape, dtype=dtype)
        if out.nbytes == 0:
            return out
        rc = self._lib.vkn_buf_download(
            self.ptr, offset, out.ctypes.data_as(ctypes.c_void_p), out.nbytes
        )
        check_rc(rc, "vkn_buf_download")
        return out

    def copy_to(self, dst: "DeviceBuffer", nbytes: int, src_off: int = 0, dst_off: int = 0) -> None:
        if nbytes == 0:
            return
        rc = self._lib.vkn_buf_copy(self.ptr, src_off, dst.ptr, dst_off, nbytes)
        check_rc(rc, "vkn_buf_copy")

    def __del__(self) -> None:
        ptr = getattr(self, "ptr", None)
        if ptr and _handle._initialized:
            self.ptr = None
            try:
                self._lib.vkn_buf_free(ptr)
            except Exception:
                pass


def i64_array(values) -> ctypes.Array:
    arr_t = ctypes.c_int64 * len(values)
    return arr_t(*values)


def i32_array(values) -> ctypes.Array:
    arr_t = ctypes.c_int * len(values)
    return arr_t(*values)
