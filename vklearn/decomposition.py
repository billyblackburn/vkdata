"""GPU Principal Component Analysis (PCA) via ggml-vulkan covariance GEMM."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_2d


class PCA:
    """Principal Component Analysis using GPU covariance GEMM + exact float64 eigensolve."""

    def __init__(self, n_components: Optional[int] = None) -> None:
        self.n_components = n_components
        self.mean_: Optional[vnp.ndarray] = None
        self.components_: Optional[vnp.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "PCA":
        Xg = to_gpu_2d(X)
        n, d = Xg.shape
        k = d if self.n_components is None else min(int(self.n_components), d)

        self.mean_ = vnp.mean(Xg, axis=0)
        Xc = Xg - self.mean_

        # Covariance matrix on GPU via ggml-vulkan GEMM
        cov = (Xc.T @ Xc).to_numpy().astype(np.float64) / max(1, n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        # Deterministic sign convention (largest abs entry per component is positive)
        comps = eigvecs[:, :k].T.copy()  # (k, d)
        for i in range(k):
            j = int(np.argmax(np.abs(comps[i])))
            if comps[i, j] < 0:
                comps[i] *= -1.0

        total_var = float(np.sum(eigvals))
        self.components_ = vnp.array(comps.astype(np.float32), dtype=np.float32)
        self.explained_variance_ = eigvals[:k].astype(np.float32)
        self.explained_variance_ratio_ = (eigvals[:k] / total_var).astype(np.float32) if total_var > 0 else np.zeros((k,), dtype=np.float32)
        return self

    def transform(self, X: Any) -> vnp.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA must be fitted before transform()")
        Xg = to_gpu_2d(X)
        return (Xg - self.mean_) @ self.components_.T

    def fit_transform(self, X: Any, y: Any = None) -> vnp.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Z: Any) -> vnp.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA must be fitted before inverse_transform()")
        Zg = to_gpu_2d(Z)
        return (Zg @ self.components_) + self.mean_
