"""Internal GPU array coercion helpers (zero-copy for vknumpy & vkpandas)."""

from __future__ import annotations

from typing import Any
import numpy as np
import vknumpy as vnp


def to_gpu_2d(X: Any) -> vnp.ndarray:
    """Convert vkpandas.DataFrame, vknumpy.ndarray, or array-like to contiguous 2D float32 GPU ndarray."""
    if isinstance(X, vnp.ndarray):
        xf = X if X.dtype == np.float32 else X.astype(np.float32)
        if xf.ndim == 1:
            return xf.reshape(-1, 1)
        if xf.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {xf.shape}")
        return xf

    # vkpandas DataFrame (stack GPU columns directly in VRAM)
    if hasattr(X, "columns") and hasattr(X, "_columns"):
        cols = [X[c].values.astype(np.float32) for c in X.columns]
        if not cols:
            return vnp.empty((0, 0), dtype=np.float32)
        return vnp.stack(cols, axis=1)

    # vkpandas Series
    if hasattr(X, "values") and isinstance(X.values, vnp.ndarray):
        return X.values.astype(np.float32).reshape(-1, 1)

    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return vnp.array(arr, dtype=np.float32)


def to_gpu_1d(y: Any, dtype: Any = np.float32) -> vnp.ndarray:
    """Convert vkpandas.Series, vknumpy.ndarray, or array-like to 1D GPU ndarray."""
    if isinstance(y, vnp.ndarray):
        return (y if y.dtype == np.dtype(dtype) else y.astype(dtype)).ravel()
    if hasattr(y, "values") and isinstance(y.values, vnp.ndarray):
        v = y.values
        return (v if v.dtype == np.dtype(dtype) else v.astype(dtype)).ravel()
    return vnp.array(np.asarray(y, dtype=dtype).ravel(), dtype=dtype)
