"""GPU pairwise spatial distances (scipy.spatial.distance compatible) accelerated via Vulkan GEMM."""

from __future__ import annotations

from typing import Any

import numpy as np
import vknumpy as vnp

from .special import _to_f32


def cdist(XA: Any, XB: Any, metric: str = "euclidean") -> vnp.ndarray:
    """Compute pairwise distances between rows of XA (M, D) and XB (N, D) on Vulkan GPU."""
    a = _to_f32(XA)
    b = _to_f32(XB)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    m = metric.lower()
    if m in ("euclidean", "sqeuclidean"):
        # Center for numerical stability with large coordinates
        mean = vnp.mean(a, axis=0, keepdims=True)
        ac = a - mean
        bc = b - mean
        a2 = vnp.sum(ac * ac, axis=1, keepdims=True)
        b2 = vnp.sum(bc * bc, axis=1, keepdims=True).T
        d2 = vnp.maximum(a2 + b2 - 2.0 * (ac @ bc.T), 0.0)
        return d2 if m == "sqeuclidean" else vnp.sqrt(d2)

    if m == "cosine":
        an = a / vnp.maximum(vnp.sqrt(vnp.sum(a * a, axis=1, keepdims=True)), 1e-12)
        bn = b / vnp.maximum(vnp.sqrt(vnp.sum(b * b, axis=1, keepdims=True)), 1e-12)
        return vnp.clip(1.0 - (an @ bn.T), 0.0, 2.0)

    if m in ("cityblock", "manhattan"):
        # Broadcast (M, 1, D) - (1, N, D)
        diff = vnp.abs(a.reshape(a.shape[0], 1, a.shape[1]) - b.reshape(1, b.shape[0], b.shape[1]))
        return vnp.sum(diff, axis=2)

    if m == "chebyshev":
        diff = vnp.abs(a.reshape(a.shape[0], 1, a.shape[1]) - b.reshape(1, b.shape[0], b.shape[1]))
        return vnp.max(diff, axis=2)

    raise ValueError(f"Unsupported metric={metric!r}; expected euclidean, sqeuclidean, cosine, cityblock, or chebyshev")


def pdist(X: Any, metric: str = "euclidean") -> vnp.ndarray:
    """Condensed pairwise distance vector for rows of X (N, D) on Vulkan GPU."""
    D = cdist(X, X, metric=metric).to_numpy()
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return vnp.array(D[iu].astype(np.float32), dtype=np.float32)


def squareform(v: Any) -> vnp.ndarray:
    """Convert between condensed distance vector and square distance matrix."""
    arr = _to_f32(v).to_numpy()
    if arr.ndim == 2:
        iu = np.triu_indices(arr.shape[0], k=1)
        return vnp.array(arr[iu].astype(np.float32), dtype=np.float32)
    m = len(arr)
    n = int(round(0.5 * (1.0 + np.sqrt(1.0 + 8.0 * m))))
    mat = np.zeros((n, n), dtype=np.float32)
    iu = np.triu_indices(n, k=1)
    mat[iu] = arr
    mat[(iu[1], iu[0])] = arr
    return vnp.array(mat, dtype=np.float32)


class _DistanceModule:
    cdist = staticmethod(cdist)
    pdist = staticmethod(pdist)
    squareform = staticmethod(squareform)


distance = _DistanceModule()
