"""Array creation routines and random submodule for vknumpy."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as _np

from ._backend import (
    NATIVE_FLOAT_DTYPES,
    NP_TO_VKN_DTYPE,
    VKN_TRI_LOWER,
    VKN_TRI_LOWER_DIAG,
    VKN_TRI_UPPER,
    VKN_TRI_UPPER_DIAG,
    check_rc,
    i64_array,
    lib,
)
from .ndarray import ndarray


def _norm_shape(shape: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(d) for d in shape)


def array(obj: Any, dtype: Any = None, copy: bool = True) -> ndarray:
    """Create a Vulkan-backed ndarray from a Python sequence or NumPy array."""
    if isinstance(obj, ndarray):
        if dtype is not None:
            return obj.astype(dtype, copy=copy)
        return obj.copy() if copy else obj

    if isinstance(obj, _np.ndarray):
        arr = obj if dtype is None else obj.astype(dtype)
        return ndarray._from_numpy(arr)

    arr = _np.array(obj, dtype=dtype)
    # Default untyped Python float lists/scalars to float32 for native GPU execution
    if dtype is None and arr.dtype == _np.float64:
        arr = arr.astype(_np.float32)
    return ndarray._from_numpy(arr)


def asarray(a: Any, dtype: Any = None) -> ndarray:
    """Convert input to a Vulkan-backed ndarray without copying if already compatible."""
    if isinstance(a, ndarray):
        if dtype is None or _np.dtype(dtype) == a.dtype:
            return a
        return a.astype(dtype)
    return array(a, dtype=dtype, copy=False)


def empty(shape: Union[int, Sequence[int]], dtype: Any = _np.float32) -> ndarray:
    return ndarray._empty_uninit(_norm_shape(shape), dtype)


def _gpu_fill(shape: Tuple[int, ...], value: float, dtype: _np.dtype) -> ndarray:
    out = ndarray._empty_uninit(shape, dtype)
    if out.size > 0:
        rc = lib().vkn_fill(out._buf.ptr, 0, out.size, float(value), NP_TO_VKN_DTYPE[dtype])
        check_rc(rc, "vkn_fill")
    return out


def zeros(shape: Union[int, Sequence[int]], dtype: Any = _np.float32) -> ndarray:
    shp = _norm_shape(shape)
    dt = _np.dtype(dtype)
    if dt in NATIVE_FLOAT_DTYPES:
        return _gpu_fill(shp, 0.0, dt)
    return ndarray._from_numpy(_np.zeros(shp, dtype=dt))


def ones(shape: Union[int, Sequence[int]], dtype: Any = _np.float32) -> ndarray:
    shp = _norm_shape(shape)
    dt = _np.dtype(dtype)
    if dt in NATIVE_FLOAT_DTYPES:
        return _gpu_fill(shp, 1.0, dt)
    return ndarray._from_numpy(_np.ones(shp, dtype=dt))


def full(shape: Union[int, Sequence[int]], fill_value: Any, dtype: Any = None) -> ndarray:
    shp = _norm_shape(shape)
    if dtype is None:
        if isinstance(fill_value, float):
            dtype = _np.float32
        elif isinstance(fill_value, int) and not isinstance(fill_value, bool):
            dtype = _np.int32
    dt = _np.dtype(dtype) if dtype is not None else _np.asarray(fill_value).dtype
    if dt in NATIVE_FLOAT_DTYPES:
        return _gpu_fill(shp, float(fill_value), dt)
    return ndarray._from_numpy(_np.full(shp, fill_value, dtype=dt))


def zeros_like(a: Any, dtype: Any = None) -> ndarray:
    arr = asarray(a)
    return zeros(arr.shape, dtype=arr.dtype if dtype is None else dtype)


def ones_like(a: Any, dtype: Any = None) -> ndarray:
    arr = asarray(a)
    return ones(arr.shape, dtype=arr.dtype if dtype is None else dtype)


def empty_like(a: Any, dtype: Any = None) -> ndarray:
    arr = asarray(a)
    return empty(arr.shape, dtype=arr.dtype if dtype is None else dtype)


def full_like(a: Any, fill_value: Any, dtype: Any = None) -> ndarray:
    arr = asarray(a)
    return full(arr.shape, fill_value, dtype=arr.dtype if dtype is None else dtype)


def arange(*args: Any, dtype: Any = None) -> ndarray:
    if len(args) == 1:
        start, stop, step = 0.0, float(args[0]), 1.0
    elif len(args) == 2:
        start, stop, step = float(args[0]), float(args[1]), 1.0
    elif len(args) == 3:
        start, stop, step = float(args[0]), float(args[1]), float(args[2])
    else:
        arr = _np.arange(*args, dtype=dtype)
        return ndarray._from_numpy(arr)

    dt = _np.dtype(dtype) if dtype is not None else (
        _np.dtype(_np.float32) if any(isinstance(a, float) for a in args) else _np.dtype(_np.int64)
    )
    if dt in NATIVE_FLOAT_DTYPES and step > 0.0 and stop > start:
        n = int(math.ceil((stop - start) / step))
        if n > 0:
            out = ndarray._empty_uninit((n,), dt)
            rc = lib().vkn_arange(start, stop, step, out._buf.ptr, 0, NP_TO_VKN_DTYPE[dt])
            check_rc(rc, "vkn_arange")
            return out

    arr = _np.arange(*args, dtype=dtype)
    if dtype is None and arr.dtype == _np.float64:
        arr = arr.astype(_np.float32)
    return ndarray._from_numpy(arr)


def linspace(
    start: float,
    stop: float,
    num: int = 50,
    endpoint: bool = True,
    dtype: Any = _np.float32,
) -> ndarray:
    return ndarray._from_numpy(_np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype))


def eye(N: int, M: Optional[int] = None, k: int = 0, dtype: Any = _np.float32) -> ndarray:
    return ndarray._from_numpy(_np.eye(N, M=M, k=k, dtype=dtype))


def identity(n: int, dtype: Any = _np.float32) -> ndarray:
    return eye(n, dtype=dtype)


def diag(v: Any, k: int = 0) -> ndarray:
    xv = asarray(v)
    if xv.ndim == 1 and k == 0 and xv.dtype in NATIVE_FLOAT_DTYPES and xv.size > 0:
        c = xv.contiguous()
        n = c.shape[0]
        out = ndarray._empty_uninit((n, n), c.dtype)
        rc = lib().vkn_diag(c._buf.ptr, c._offset, n, out._buf.ptr, 0, NP_TO_VKN_DTYPE[c.dtype])
        check_rc(rc, "vkn_diag")
        return out
    return ndarray._from_numpy(_np.diag(xv.to_numpy(), k=k))


def tril(m: Any, k: int = 0) -> ndarray:
    """Lower triangle of an array, GPU-accelerated via ggml_tri for k in (0, -1)."""
    xm = asarray(m)
    if xm.dtype in NATIVE_FLOAT_DTYPES and 2 <= xm.ndim <= 4 and k in (0, -1) and xm.size > 0:
        c = xm.contiguous()
        tri_type = VKN_TRI_LOWER_DIAG if k == 0 else VKN_TRI_LOWER
        out = ndarray._empty_uninit(c.shape, c.dtype)
        rc = lib().vkn_tri(
            c._buf.ptr, c._offset, i64_array(c.shape), c.ndim,
            tri_type, out._buf.ptr, 0, NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_tri")
        return out
    return ndarray._from_numpy(_np.tril(xm.to_numpy(), k=k))


def triu(m: Any, k: int = 0) -> ndarray:
    """Upper triangle of an array, GPU-accelerated via ggml_tri for k in (0, 1)."""
    xm = asarray(m)
    if xm.dtype in NATIVE_FLOAT_DTYPES and 2 <= xm.ndim <= 4 and k in (0, 1) and xm.size > 0:
        c = xm.contiguous()
        tri_type = VKN_TRI_UPPER_DIAG if k == 0 else VKN_TRI_UPPER
        out = ndarray._empty_uninit(c.shape, c.dtype)
        rc = lib().vkn_tri(
            c._buf.ptr, c._offset, i64_array(c.shape), c.ndim,
            tri_type, out._buf.ptr, 0, NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_tri")
        return out
    return ndarray._from_numpy(_np.triu(xm.to_numpy(), k=k))


class _RandomModule:
    """Drop-in subset of numpy.random producing float32 GPU ndarrays by default."""

    def seed(self, seed: Optional[int] = None) -> None:
        _np.random.seed(seed)

    def rand(self, *d: int, dtype: Any = _np.float32) -> ndarray:
        arr = _np.random.rand(*d)
        if isinstance(arr, _np.ndarray):
            return ndarray._from_numpy(arr.astype(dtype))
        return ndarray._from_numpy(_np.array(arr, dtype=dtype))

    def randn(self, *d: int, dtype: Any = _np.float32) -> ndarray:
        arr = _np.random.randn(*d)
        if isinstance(arr, _np.ndarray):
            return ndarray._from_numpy(arr.astype(dtype))
        return ndarray._from_numpy(_np.array(arr, dtype=dtype))

    def uniform(
        self, low: float = 0.0, high: float = 1.0, size: Any = None, dtype: Any = _np.float32
    ) -> ndarray:
        arr = _np.random.uniform(low, high, size=size)
        return ndarray._from_numpy(_np.asarray(arr, dtype=dtype))

    def normal(
        self, loc: float = 0.0, scale: float = 1.0, size: Any = None, dtype: Any = _np.float32
    ) -> ndarray:
        arr = _np.random.normal(loc, scale, size=size)
        return ndarray._from_numpy(_np.asarray(arr, dtype=dtype))

    def randint(self, low: int, high: Optional[int] = None, size: Any = None, dtype: Any = _np.int32) -> ndarray:
        arr = _np.random.randint(low, high=high, size=size, dtype=dtype)
        return ndarray._from_numpy(_np.asarray(arr, dtype=dtype))

    def default_rng(self, seed: Any = None) -> "_Generator":
        return _Generator(_np.random.default_rng(seed))


class _Generator:
    def __init__(self, rng: _np.random.Generator) -> None:
        self._rng = rng

    def random(self, size: Any = None, dtype: Any = _np.float32) -> ndarray:
        return ndarray._from_numpy(_np.asarray(self._rng.random(size=size, dtype=dtype)))

    def standard_normal(self, size: Any = None, dtype: Any = _np.float32) -> ndarray:
        return ndarray._from_numpy(_np.asarray(self._rng.standard_normal(size=size, dtype=dtype)))

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any = None, dtype: Any = _np.float32) -> ndarray:
        return ndarray._from_numpy(_np.asarray(self._rng.normal(loc, scale, size=size), dtype=dtype))

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any = None, dtype: Any = _np.float32) -> ndarray:
        return ndarray._from_numpy(_np.asarray(self._rng.uniform(low, high, size=size), dtype=dtype))


random = _RandomModule()
