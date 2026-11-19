"""Array creation routines and random submodule for vknumpy."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union
import numpy as _np

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


def zeros(shape: Union[int, Sequence[int]], dtype: Any = _np.float32) -> ndarray:
    return ndarray._from_numpy(_np.zeros(_norm_shape(shape), dtype=dtype))


def ones(shape: Union[int, Sequence[int]], dtype: Any = _np.float32) -> ndarray:
    return ndarray._from_numpy(_np.ones(_norm_shape(shape), dtype=dtype))


def full(shape: Union[int, Sequence[int]], fill_value: Any, dtype: Any = None) -> ndarray:
    if dtype is None:
        if isinstance(fill_value, float):
            dtype = _np.float32
        elif isinstance(fill_value, int) and not isinstance(fill_value, bool):
            dtype = _np.int32
    return ndarray._from_numpy(_np.full(_norm_shape(shape), fill_value, dtype=dtype))


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
    arr = v.to_numpy() if isinstance(v, ndarray) else _np.asarray(v)
    return ndarray._from_numpy(_np.diag(arr, k=k))


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
