"""GPU statistics (scipy.stats compatible): norm, zscore, pearsonr, corrcoef, skew, kurtosis."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import vknumpy as vnp

from .special import _to_f32, ndtr


class _NormDist:
    """Standard & parameterized Normal distribution evaluated on Vulkan GPU."""

    @staticmethod
    def pdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> vnp.ndarray:
        z = (_to_f32(x) - float(loc)) / float(scale)
        inv_s = float(1.0 / (math.sqrt(2.0 * math.pi) * scale))
        return vnp.exp(-0.5 * (z * z)) * inv_s

    @staticmethod
    def logpdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> vnp.ndarray:
        z = (_to_f32(x) - float(loc)) / float(scale)
        c = -0.5 * math.log(2.0 * math.pi) - math.log(float(scale))
        return -0.5 * (z * z) + float(c)

    @staticmethod
    def cdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> vnp.ndarray:
        z = (_to_f32(x) - float(loc)) / float(scale)
        return ndtr(z)


norm = _NormDist()


def zscore(a: Any, axis: Optional[int] = 0, ddof: int = 0) -> vnp.ndarray:
    """Standardize array to zero mean and unit variance on Vulkan GPU."""
    ag = _to_f32(a)
    mean = vnp.mean(ag, axis=axis, keepdims=True)
    diff = ag - mean
    n = ag.size if axis is None else ag.shape[axis]
    var = vnp.sum(diff * diff, axis=axis, keepdims=True) / float(max(1, n - ddof))
    return diff / vnp.maximum(vnp.sqrt(var), 1e-12)


def pearsonr(x: Any, y: Any) -> Tuple[float, float]:
    """Pearson correlation coefficient r (and normal-approx two-sided p-value) on GPU."""
    xg = _to_f32(x).ravel()
    yg = _to_f32(y).ravel()
    xc = xg - vnp.mean(xg)
    yc = yg - vnp.mean(yg)
    num = float(vnp.sum(xc * yc).item())
    den = math.sqrt(max(float(vnp.sum(xc * xc).item()) * float(vnp.sum(yc * yc).item()), 1e-24))
    r = max(-1.0, min(1.0, num / den))
    n = max(3, xg.size)
    # Fisher z-transform two-sided p-value
    if abs(r) >= 1.0 - 1e-12:
        p = 0.0
    else:
        z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(max(1, n - 3))
        p = float(math.erfc(abs(z) / math.sqrt(2.0)))
    return float(r), p


def corrcoef(X: Any) -> vnp.ndarray:
    """Feature-by-feature Pearson correlation matrix for X (N, D) via Vulkan GPU GEMM."""
    xg = _to_f32(X)
    xc = xg - vnp.mean(xg, axis=0, keepdims=True)
    norms = vnp.maximum(vnp.sqrt(vnp.sum(xc * xc, axis=0, keepdims=True)), 1e-12)
    xn = xc / norms
    return vnp.clip(xn.T @ xn, -1.0, 1.0)


def skew(a: Any, axis: Optional[int] = 0) -> vnp.ndarray:
    """Sample skewness on Vulkan GPU."""
    ag = _to_f32(a)
    m = vnp.mean(ag, axis=axis, keepdims=True)
    d = ag - m
    m2 = vnp.mean(d * d, axis=axis)
    m3 = vnp.mean(d * d * d, axis=axis)
    return m3 / vnp.maximum(m2 * vnp.sqrt(vnp.maximum(m2, 1e-24)), 1e-24)


def kurtosis(a: Any, axis: Optional[int] = 0, fisher: bool = True) -> vnp.ndarray:
    """Sample kurtosis (excess Fisher kurtosis by default) on Vulkan GPU."""
    ag = _to_f32(a)
    m = vnp.mean(ag, axis=axis, keepdims=True)
    d = ag - m
    d2 = d * d
    m2 = vnp.mean(d2, axis=axis)
    m4 = vnp.mean(d2 * d2, axis=axis)
    k = m4 / vnp.maximum(m2 * m2, 1e-24)
    return (k - 3.0) if fisher else k
