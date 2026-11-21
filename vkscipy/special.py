"""GPU special mathematical functions (SciPy-compatible): expit, logit, erf, erfc, ndtr, softmax, logsumexp."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import vknumpy as vnp


def _to_f32(a: Any) -> vnp.ndarray:
    if hasattr(a, "values") and isinstance(a.values, vnp.ndarray):
        arr = a.values
        return arr if arr.dtype == np.float32 else arr.astype(np.float32)
    if isinstance(a, vnp.ndarray):
        return a if a.dtype == np.float32 else a.astype(np.float32)
    return vnp.asarray(a, dtype=np.float32)


def expit(x: Any) -> vnp.ndarray:
    """Logistic sigmoid 1 / (1 + exp(-x)) on Vulkan GPU."""
    xg = _to_f32(x)
    return 1.0 / (1.0 + vnp.exp(-vnp.clip(xg, -40.0, 40.0)))


def logit(p: Any) -> vnp.ndarray:
    """Log-odds log(p / (1 - p)) on Vulkan GPU."""
    pg = vnp.clip(_to_f32(p), 1e-7, 1.0 - 1e-7)
    return vnp.log(pg / (1.0 - pg))


def erf(x: Any) -> vnp.ndarray:
    """Error function erf(x) on Vulkan GPU (Abramowitz & Stegun 7.1.26, max error < 1.5e-7)."""
    xg = _to_f32(x)
    ax = vnp.abs(xg)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    y = 1.0 - poly * vnp.exp(-(ax * ax))
    return vnp.where(xg >= 0.0, y, -y)


def erfc(x: Any) -> vnp.ndarray:
    """Complementary error function 1 - erf(x) on Vulkan GPU."""
    return 1.0 - erf(x)


def ndtr(x: Any) -> vnp.ndarray:
    """Standard normal cumulative distribution function Phi(x) on Vulkan GPU."""
    return 0.5 * (1.0 + erf(_to_f32(x) * float(1.0 / math.sqrt(2.0))))


def logsumexp(x: Any, axis: Optional[int] = None, keepdims: bool = False) -> vnp.ndarray:
    """Numerically stable log(sum(exp(x))) on Vulkan GPU."""
    xg = _to_f32(x)
    m = vnp.max(xg, axis=axis, keepdims=True)
    s = vnp.log(vnp.sum(vnp.exp(xg - m), axis=axis, keepdims=True)) + m
    if not keepdims:
        if axis is None:
            return s.reshape(())
        return vnp.squeeze(s, axis=axis) if hasattr(vnp, "squeeze") else vnp.array(np.squeeze(s.to_numpy(), axis=axis), dtype=np.float32)
    return s


def softmax(x: Any, axis: int = -1) -> vnp.ndarray:
    """Numerically stable softmax on Vulkan GPU."""
    xg = _to_f32(x)
    m = vnp.max(xg, axis=axis, keepdims=True)
    ex = vnp.exp(xg - m)
    return ex / vnp.sum(ex, axis=axis, keepdims=True)
