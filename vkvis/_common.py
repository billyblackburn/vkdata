"""Shared GPU initialization, affinity calibration, and optimizers for vkvis."""

from __future__ import annotations

from typing import Tuple

import numpy as _np
import vknumpy as vnp


def pca_project_gpu(X_np: _np.ndarray, n_components: int, seed: int = 42) -> _np.ndarray:
    """Fast GPU PCA projection of ``X_np`` (N, D) -> (N, n_components) via vknumpy."""
    n, d = X_np.shape
    k = min(int(n_components), n, d)
    X_gpu = vnp.array(X_np, dtype=_np.float32)
    mean = vnp.mean(X_gpu, axis=0).reshape(1, d)
    Xc = X_gpu - mean

    if d <= 512:
        # Exact covariance on GPU: (D, D) = Xc.T @ Xc
        cov = (Xc.T.contiguous() @ Xc).to_numpy() / max(1, n - 1)
        eigvals, eigvecs = _np.linalg.eigh(cov.astype(_np.float64))
        top_v = eigvecs[:, ::-1][:, :k].astype(_np.float32)
        Y = (Xc @ vnp.array(top_v, dtype=_np.float32)).to_numpy()
    else:
        # Randomized GPU PCA (Halko et al.) for very high-dimensional inputs
        rng = _np.random.default_rng(seed)
        r = min(d, k + 12)
        omega = vnp.array(rng.standard_normal((d, r)).astype(_np.float32))
        Z = Xc @ omega
        for _ in range(2):
            Z = Xc @ (Xc.T.contiguous() @ Z)
        Q, _ = _np.linalg.qr(Z.to_numpy().astype(_np.float64))
        Q_gpu = vnp.array(Q.astype(_np.float32))
        B = (Q_gpu.T.contiguous() @ Xc).to_numpy().astype(_np.float64)
        U_b, S_b, _ = _np.linalg.svd(B, full_matrices=False)
        Y = (Q @ (U_b[:, :k] * S_b[:k])).astype(_np.float32)

    if k < n_components:
        pad = _np.zeros((n, n_components - k), dtype=_np.float32)
        Y = _np.concatenate([Y, pad], axis=1)
    return _np.ascontiguousarray(Y, dtype=_np.float32)


def init_embedding(
    X_np: _np.ndarray,
    n_components: int = 2,
    init: str = "pca",
    scale: float = 10.0,
    seed: int = 42,
) -> _np.ndarray:
    """Initialize low-dimensional coordinates ``Y`` of shape ``(N, n_components)``."""
    n = X_np.shape[0]
    rng = _np.random.default_rng(seed)
    if isinstance(init, _np.ndarray):
        Y = _np.ascontiguousarray(init, dtype=_np.float32)
        if Y.shape != (n, n_components):
            raise ValueError(f"Custom init has shape {Y.shape}, expected {(n, n_components)}")
        return Y
    if init == "pca":
        Y = pca_project_gpu(X_np, n_components=n_components, seed=seed)
        std = float(_np.std(Y[:, 0])) + 1e-8
        Y = (Y / std) * float(scale)
        # Add tiny jitter to break exact ties
        Y += rng.normal(0.0, float(scale) * 1e-4, size=Y.shape).astype(_np.float32)
        return Y.astype(_np.float32)
    if init == "random":
        return rng.normal(0.0, float(scale), size=(n, n_components)).astype(_np.float32)
    raise ValueError(f"Unknown init='{init}'. Use 'pca', 'random', or an ndarray.")


def smooth_knn_dist(distances: _np.ndarray, k: int, n_iter: int = 48) -> Tuple[_np.ndarray, _np.ndarray]:
    """Vectorized binary search for UMAP's local connectivity ``rho`` and bandwidth ``sigma``."""
    d = _np.asarray(distances, dtype=_np.float32)
    rho = _np.maximum(d[:, 0], 0.0)
    target = _np.float32(_np.log2(max(float(k), 2.0)))

    lo = _np.zeros(d.shape[0], dtype=_np.float32)
    hi = _np.full(d.shape[0], _np.float32(1e6), dtype=_np.float32)
    mid = _np.ones(d.shape[0], dtype=_np.float32)

    diff = _np.maximum(d - rho[:, None], 0.0)
    for _ in range(n_iter):
        val = _np.sum(_np.exp(-diff / _np.maximum(mid[:, None], 1e-8)), axis=1)
        too_high = val > target
        hi = _np.where(too_high, mid, hi)
        lo = _np.where(too_high, lo, mid)
        mid = _np.where(hi < 1e5, 0.5 * (lo + hi), mid * 2.0)

    mean_d = _np.mean(d, axis=1)
    sigma = _np.maximum(mid, 1e-3 * _np.maximum(mean_d, 1e-4))
    return rho, sigma


def umap_fuzzy_weights(distances: _np.ndarray, k: int) -> _np.ndarray:
    """Compute directed UMAP fuzzy membership weights ``exp(-max(0, d - rho) / sigma)``."""
    rho, sigma = smooth_knn_dist(distances, k=k)
    diff = _np.maximum(distances - rho[:, None], 0.0)
    w = _np.exp(-diff / _np.maximum(sigma[:, None], 1e-8))
    return _np.clip(w, 1e-4, 1.0).astype(_np.float32)


def entropic_affinities(distances: _np.ndarray, perplexity: float = 30.0, n_iter: int = 50) -> _np.ndarray:
    """Vectorized binary search for Gaussian precisions ``beta_i`` matching ``perplexity``."""
    d2 = _np.asarray(distances, dtype=_np.float64) ** 2
    n = d2.shape[0]
    target_h = _np.log(max(float(perplexity), 2.0))

    lo = _np.zeros(n, dtype=_np.float64)
    hi = _np.full(n, 1e6, dtype=_np.float64)
    beta = _np.ones(n, dtype=_np.float64)

    for _ in range(n_iter):
        logits = -d2 * beta[:, None]
        logits -= _np.max(logits, axis=1, keepdims=True)
        p = _np.exp(logits)
        s = _np.sum(p, axis=1, keepdims=True)
        p = p / _np.maximum(s, 1e-12)
        h = -_np.sum(p * _np.log(_np.maximum(p, 1e-12)), axis=1)
        too_high = h > target_h
        lo = _np.where(too_high, beta, lo)
        hi = _np.where(too_high, hi, beta)
        beta = _np.where(hi < 1e5, 0.5 * (lo + hi), beta * 2.0)

    logits = -d2 * beta[:, None]
    logits -= _np.max(logits, axis=1, keepdims=True)
    p = _np.exp(logits)
    p = p / _np.maximum(_np.sum(p, axis=1, keepdims=True), 1e-12)
    return p.astype(_np.float32)


def fit_umap_ab(spread: float = 1.0, min_dist: float = 0.1) -> Tuple[float, float]:
    """Fit UMAP curve parameters ``(a, b)`` for ``1 / (1 + a * x^(2b))`` without SciPy."""
    xv = _np.linspace(0.0, 3.0 * spread, 300, dtype=_np.float64)
    yv = _np.where(xv < min_dist, 1.0, _np.exp(-(xv - min_dist) / max(spread, 1e-6)))

    # Gauss-Newton / damped gradient refinement from canonical initial guess
    a, b = 1.577, 0.895
    eps = 1e-9
    for _ in range(80):
        x2b = _np.power(_np.maximum(xv, eps), 2.0 * b)
        denom = 1.0 + a * x2b
        f = 1.0 / denom
        r = f - yv
        # Partial derivatives
        df_da = -x2b / (denom * denom)
        df_db = -2.0 * a * x2b * _np.log(_np.maximum(xv, eps)) / (denom * denom)
        J = _np.column_stack([df_da, df_db])
        H = J.T @ J + 1e-6 * _np.eye(2)
        g = J.T @ r
        step = _np.linalg.solve(H, g)
        a = float(_np.clip(a - step[0], 1e-3, 1e3))
        b = float(_np.clip(b - step[1], 0.1, 5.0))
        if float(_np.linalg.norm(step)) < 1e-7:
            break
    return float(a), float(b)


class GPUAdam:
    """Pure Vulkan GPU Adam optimizer operating on ``vknumpy.ndarray`` tensors."""

    def __init__(self, shape: Tuple[int, ...], lr: float = 1.0, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-7) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = vnp.zeros(shape, dtype=_np.float32)
        self.v = vnp.zeros(shape, dtype=_np.float32)
        self.t = 0

    def step(self, Y: vnp.ndarray, grad: vnp.ndarray, lr_scale: float = 1.0) -> vnp.ndarray:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        self.m = self.m * b1 + grad * (1.0 - b1)
        self.v = self.v * b2 + (grad * grad) * (1.0 - b2)
        bc1 = 1.0 - (b1 ** self.t)
        bc2 = 1.0 - (b2 ** self.t)
        m_hat = self.m * (1.0 / bc1)
        v_hat = self.v * (1.0 / bc2)
        lr_eff = self.lr * float(lr_scale)
        return Y - m_hat * (lr_eff / (vnp.sqrt(v_hat) + self.eps))
