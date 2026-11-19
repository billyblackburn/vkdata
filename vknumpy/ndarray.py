"""Core GPU-backed ndarray class for vknumpy."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as _np

from ._backend import (
    DeviceBuffer,
    NATIVE_FLOAT_DTYPES,
    NP_TO_VKN_DTYPE,
    check_rc,
    i32_array,
    i64_array,
    lib,
)


def _prod(shape: Tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


class ndarray:
    """N-dimensional array stored in Vulkan device memory via ggml-vulkan."""

    __slots__ = ("_buf", "_shape", "_dtype", "_offset", "_transposed_last2")

    def __init__(
        self,
        buf: DeviceBuffer,
        shape: Tuple[int, ...],
        dtype: Union[_np.dtype, type, str],
        offset: int = 0,
        transposed_last2: bool = False,
    ) -> None:
        self._buf = buf
        self._shape = tuple(int(d) for d in shape)
        self._dtype = _np.dtype(dtype)
        self._offset = int(offset)
        self._transposed_last2 = bool(transposed_last2)

    @classmethod
    def _from_numpy(cls, arr: _np.ndarray) -> "ndarray":
        c_arr = _np.ascontiguousarray(arr)
        buf = DeviceBuffer(c_arr.nbytes)
        buf.upload(c_arr)
        return cls(buf, c_arr.shape, c_arr.dtype)

    @classmethod
    def _empty_uninit(cls, shape: Tuple[int, ...], dtype: Union[_np.dtype, type, str]) -> "ndarray":
        dt = _np.dtype(dtype)
        shp = tuple(int(d) for d in shape)
        nbytes = _prod(shp) * dt.itemsize
        buf = DeviceBuffer(nbytes)
        return cls(buf, shp, dt)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return _prod(self._shape)

    @property
    def dtype(self) -> _np.dtype:
        return self._dtype

    @property
    def itemsize(self) -> int:
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self._dtype.itemsize

    @property
    def device(self) -> str:
        from .device import is_vulkan_available
        return "vulkan" if is_vulkan_available() else "cpu"

    @property
    def _storage_shape(self) -> Tuple[int, ...]:
        if self._transposed_last2 and len(self._shape) >= 2:
            return self._shape[:-2] + (self._shape[-1], self._shape[-2])
        return self._shape

    @property
    def T(self) -> "ndarray":
        if self.ndim < 2:
            return self
        if self.ndim == 2:
            return ndarray(
                self._buf,
                (self._shape[1], self._shape[0]),
                self._dtype,
                offset=self._offset,
                transposed_last2=not self._transposed_last2,
            )
        return self.transpose()

    # ------------------------------------------------------------------
    # Host / Device transfers & conversions
    # ------------------------------------------------------------------
    def contiguous(self) -> "ndarray":
        """Return a C-contiguous GPU ndarray (no-op if already contiguous)."""
        if not self._transposed_last2:
            return self

        if self._dtype in NP_TO_VKN_DTYPE and 2 <= self.ndim <= 4 and self.size > 0:
            storage_shape = self._storage_shape
            perm = list(range(self.ndim))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            out = ndarray._empty_uninit(self._shape, self._dtype)
            rc = lib().vkn_transpose_cont(
                self._buf.ptr,
                self._offset,
                i64_array(storage_shape),
                i32_array(perm),
                self.ndim,
                out._buf.ptr,
                0,
                NP_TO_VKN_DTYPE[self._dtype],
            )
            check_rc(rc, "vkn_transpose_cont")
            return out

        return ndarray._from_numpy(self.to_numpy())

    def to_numpy(self) -> _np.ndarray:
        """Download array from Vulkan device memory to a NumPy ndarray."""
        raw = self._buf.download(self._storage_shape, self._dtype, offset=self._offset)
        if self._transposed_last2 and self.ndim >= 2:
            return _np.ascontiguousarray(_np.swapaxes(raw, -1, -2))
        return raw

    get = to_numpy

    def __array__(self, dtype: Any = None, copy: Any = None) -> _np.ndarray:
        arr = self.to_numpy()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def item(self) -> Any:
        return self.to_numpy().item()

    def tolist(self) -> Any:
        return self.to_numpy().tolist()

    def __float__(self) -> float:
        return float(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __bool__(self) -> bool:
        return bool(self.item())

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    def __repr__(self) -> str:
        np_repr = repr(self.to_numpy())
        if np_repr.startswith("array("):
            np_repr = "vk" + np_repr
        return np_repr

    def __str__(self) -> str:
        return str(self.to_numpy())

    # ------------------------------------------------------------------
    # Shape & type transformations
    # ------------------------------------------------------------------
    def copy(self) -> "ndarray":
        c = self.contiguous()
        out = ndarray._empty_uninit(c._shape, c._dtype)
        c._buf.copy_to(out._buf, c.nbytes, src_off=c._offset, dst_off=0)
        return out

    def astype(self, dtype: Union[_np.dtype, type, str], copy: bool = True) -> "ndarray":
        dt = _np.dtype(dtype)
        if dt == self._dtype and not copy:
            return self
        if dt == self._dtype:
            return self.copy()
        c = self.contiguous()
        if c._dtype in NATIVE_FLOAT_DTYPES and dt in NATIVE_FLOAT_DTYPES and c.size > 0:
            out = ndarray._empty_uninit(c._shape, dt)
            rc = lib().vkn_cast(
                c._buf.ptr,
                c._offset,
                NP_TO_VKN_DTYPE[c._dtype],
                c.size,
                out._buf.ptr,
                0,
                NP_TO_VKN_DTYPE[dt],
            )
            check_rc(rc, "vkn_cast")
            return out
        return ndarray._from_numpy(c.to_numpy().astype(dt))

    def reshape(self, *shape: Any) -> "ndarray":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(int(d) for d in shape[0])
        else:
            new_shape = tuple(int(d) for d in shape)

        # Resolve -1 dimension
        neg_idx = -1
        known_prod = 1
        for i, d in enumerate(new_shape):
            if d == -1:
                if neg_idx != -1:
                    raise ValueError("can only specify one unknown dimension")
                neg_idx = i
            elif d < 0:
                raise ValueError(f"negative dimension {d} not allowed")
            else:
                known_prod *= d

        if neg_idx != -1:
            if known_prod == 0 or self.size % known_prod != 0:
                raise ValueError(f"cannot reshape array of size {self.size} into shape {new_shape}")
            lst = list(new_shape)
            lst[neg_idx] = self.size // known_prod
            new_shape = tuple(lst)
        elif _prod(new_shape) != self.size:
            raise ValueError(f"cannot reshape array of size {self.size} into shape {new_shape}")

        c = self.contiguous()
        return ndarray(c._buf, new_shape, c._dtype, offset=c._offset, transposed_last2=False)

    def flatten(self) -> "ndarray":
        return self.copy().reshape(-1)

    def ravel(self) -> "ndarray":
        return self.reshape(-1)

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> "ndarray":
        if axis is None:
            new_shape = tuple(d for d in self._shape if d != 1)
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            norm = set()
            for a in axes:
                a_norm = a + self.ndim if a < 0 else a
                if a_norm < 0 or a_norm >= self.ndim:
                    raise ValueError(f"axis {a} is out of bounds for array of dimension {self.ndim}")
                if self._shape[a_norm] != 1:
                    raise ValueError(f"cannot select an axis to squeeze out which has size not equal to one")
                norm.add(a_norm)
            new_shape = tuple(d for i, d in enumerate(self._shape) if i not in norm)
        return self.reshape(new_shape)

    def swapaxes(self, axis1: int, axis2: int) -> "ndarray":
        a1 = axis1 + self.ndim if axis1 < 0 else axis1
        a2 = axis2 + self.ndim if axis2 < 0 else axis2
        if a1 < 0 or a1 >= self.ndim or a2 < 0 or a2 >= self.ndim:
            raise ValueError("axis out of bounds")
        if a1 == a2:
            return self
        # Fast zero-copy view when swapping the last two axes!
        if {a1, a2} == {self.ndim - 2, self.ndim - 1}:
            new_shape = self._shape[:-2] + (self._shape[-1], self._shape[-2])
            return ndarray(
                self._buf,
                new_shape,
                self._dtype,
                offset=self._offset,
                transposed_last2=not self._transposed_last2,
            )
        perm = list(range(self.ndim))
        perm[a1], perm[a2] = perm[a2], perm[a1]
        return self.transpose(perm)

    def transpose(self, *axes: Any) -> "ndarray":
        if len(axes) == 0 or (len(axes) == 1 and axes[0] is None):
            perm = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(int(a) for a in axes[0])
        else:
            perm = tuple(int(a) for a in axes)

        perm = tuple(a + self.ndim if a < 0 else a for a in perm)
        if len(perm) != self.ndim or sorted(perm) != list(range(self.ndim)):
            raise ValueError(f"axes don't match array: {perm}")

        if perm == tuple(range(self.ndim)):
            return self

        # Zero-copy view for swapping last two axes
        if self.ndim >= 2 and perm == tuple(range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2):
            return self.swapaxes(-1, -2)

        c = self.contiguous()
        out_shape = tuple(c._shape[p] for p in perm)
        if c._dtype in NP_TO_VKN_DTYPE and 2 <= c.ndim <= 4 and c.size > 0:
            out = ndarray._empty_uninit(out_shape, c._dtype)
            rc = lib().vkn_transpose_cont(
                c._buf.ptr,
                c._offset,
                i64_array(c._shape),
                i32_array(perm),
                c.ndim,
                out._buf.ptr,
                0,
                NP_TO_VKN_DTYPE[c._dtype],
            )
            check_rc(rc, "vkn_transpose_cont")
            return out

        return ndarray._from_numpy(_np.transpose(c.to_numpy(), perm))

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        # Zero-copy fast path for contiguous leading-axis integer/slice indexing
        if not self._transposed_last2 and self.ndim >= 1:
            if isinstance(key, int):
                idx = key + self._shape[0] if key < 0 else key
                if 0 <= idx < self._shape[0]:
                    row_elems = _prod(self._shape[1:])
                    off = self._offset + idx * row_elems * self.itemsize
                    return ndarray(self._buf, self._shape[1:], self._dtype, offset=off)
            elif isinstance(key, slice) and (key.step is None or key.step == 1):
                start, stop, _ = key.indices(self._shape[0])
                row_elems = _prod(self._shape[1:])
                off = self._offset + start * row_elems * self.itemsize
                new_shape = (max(0, stop - start),) + self._shape[1:]
                return ndarray(self._buf, new_shape, self._dtype, offset=off)

        res = self.to_numpy()[key]
        if isinstance(res, _np.ndarray):
            return ndarray._from_numpy(res)
        return res

    def __setitem__(self, key: Any, value: Any) -> None:
        arr = self.to_numpy()
        val = value.to_numpy() if isinstance(value, ndarray) else value
        arr[key] = val
        if self._transposed_last2 and self.ndim >= 2:
            arr = _np.ascontiguousarray(_np.swapaxes(arr, -1, -2))
        self._buf.upload(arr, offset=self._offset)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def __matmul__(self, other: Any) -> "ndarray":
        from .ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other: Any) -> "ndarray":
        from .ops import matmul
        return matmul(other, self)

    def __add__(self, other: Any) -> "ndarray":
        from .ops import add
        return add(self, other)

    def __radd__(self, other: Any) -> "ndarray":
        from .ops import add
        return add(other, self)

    def __sub__(self, other: Any) -> "ndarray":
        from .ops import subtract
        return subtract(self, other)

    def __rsub__(self, other: Any) -> "ndarray":
        from .ops import subtract
        return subtract(other, self)

    def __mul__(self, other: Any) -> "ndarray":
        from .ops import multiply
        return multiply(self, other)

    def __rmul__(self, other: Any) -> "ndarray":
        from .ops import multiply
        return multiply(other, self)

    def __truediv__(self, other: Any) -> "ndarray":
        from .ops import divide
        return divide(self, other)

    def __rtruediv__(self, other: Any) -> "ndarray":
        from .ops import divide
        return divide(other, self)

    def __floordiv__(self, other: Any) -> "ndarray":
        from .ops import floor_divide
        return floor_divide(self, other)

    def __rfloordiv__(self, other: Any) -> "ndarray":
        from .ops import floor_divide
        return floor_divide(other, self)

    def __mod__(self, other: Any) -> "ndarray":
        from .ops import remainder
        return remainder(self, other)

    def __rmod__(self, other: Any) -> "ndarray":
        from .ops import remainder
        return remainder(other, self)

    def __pow__(self, other: Any) -> "ndarray":
        from .ops import power
        return power(self, other)

    def __rpow__(self, other: Any) -> "ndarray":
        from .ops import power
        return power(other, self)

    def __neg__(self) -> "ndarray":
        from .ops import negative
        return negative(self)

    def __pos__(self) -> "ndarray":
        return self.copy()

    def __abs__(self) -> "ndarray":
        from .ops import abs as vk_abs
        return vk_abs(self)

    def __eq__(self, other: Any) -> "ndarray":
        from .ops import equal
        return equal(self, other)

    def __ne__(self, other: Any) -> "ndarray":
        from .ops import not_equal
        return not_equal(self, other)

    def __lt__(self, other: Any) -> "ndarray":
        from .ops import less
        return less(self, other)

    def __le__(self, other: Any) -> "ndarray":
        from .ops import less_equal
        return less_equal(self, other)

    def __gt__(self, other: Any) -> "ndarray":
        from .ops import greater
        return greater(self, other)

    def __ge__(self, other: Any) -> "ndarray":
        from .ops import greater_equal
        return greater_equal(self, other)

    # ------------------------------------------------------------------
    # Reduction & math methods
    # ------------------------------------------------------------------
    def sum(self, axis: Any = None, keepdims: bool = False) -> Any:
        from .ops import sum as vk_sum
        return vk_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> Any:
        from .ops import mean as vk_mean
        return vk_mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis: Any = None, keepdims: bool = False) -> Any:
        from .ops import max as vk_max
        return vk_max(self, axis=axis, keepdims=keepdims)

    def min(self, axis: Any = None, keepdims: bool = False) -> Any:
        from .ops import min as vk_min
        return vk_min(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis: Any = None, keepdims: bool = False) -> Any:
        from .ops import argmax as vk_argmax
        return vk_argmax(self, axis=axis, keepdims=keepdims)

    def argmin(self, axis: Any = None, keepdims: bool = False) -> Any:
        from .ops import argmin as vk_argmin
        return vk_argmin(self, axis=axis, keepdims=keepdims)

    def std(self, axis: Any = None, ddof: int = 0, keepdims: bool = False) -> Any:
        from .ops import std as vk_std
        return vk_std(self, axis=axis, ddof=ddof, keepdims=keepdims)

    def var(self, axis: Any = None, ddof: int = 0, keepdims: bool = False) -> Any:
        from .ops import var as vk_var
        return vk_var(self, axis=axis, ddof=ddof, keepdims=keepdims)

    def dot(self, b: Any) -> Any:
        from .ops import dot as vk_dot
        return vk_dot(self, b)

    def clip(self, a_min: Any = None, a_max: Any = None) -> "ndarray":
        from .ops import clip as vk_clip
        return vk_clip(self, a_min, a_max)

    # ------------------------------------------------------------------
    # NumPy protocol interoperability
    # ------------------------------------------------------------------
    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__" or kwargs:
            np_inputs = [x.to_numpy() if isinstance(x, ndarray) else x for x in inputs]
            res = getattr(ufunc, method)(*np_inputs, **kwargs)
            return ndarray._from_numpy(res) if isinstance(res, _np.ndarray) else res

        from . import ops
        _UFUNC_MAP = {
            _np.add: ops.add,
            _np.subtract: ops.subtract,
            _np.multiply: ops.multiply,
            _np.divide: ops.divide,
            _np.true_divide: ops.divide,
            _np.power: ops.power,
            _np.negative: ops.negative,
            _np.absolute: ops.abs,
            _np.square: ops.square,
            _np.sqrt: ops.sqrt,
            _np.exp: ops.exp,
            _np.log: ops.log,
            _np.sin: ops.sin,
            _np.cos: ops.cos,
            _np.tanh: ops.tanh,
            _np.matmul: ops.matmul,
            _np.maximum: ops.maximum,
            _np.minimum: ops.minimum,
        }
        fn = _UFUNC_MAP.get(ufunc)
        if fn is not None:
            return fn(*inputs)

        np_inputs = [x.to_numpy() if isinstance(x, ndarray) else x for x in inputs]
        res = ufunc(*np_inputs)
        return ndarray._from_numpy(res) if isinstance(res, _np.ndarray) else res
