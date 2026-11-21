"""GPU numerical optimization (scipy.optimize compatible): curve_fit & minimize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import vknumpy as vnp

from .special import _to_f32


@dataclass
class OptimizeResult:
    x: np.ndarray
    fun: float
    nit: int
    success: bool


def curve_fit(
    f: Callable[..., vnp.ndarray],
    xdata: Any,
    ydata: Any,
    p0: Any,
    maxfev: int = 80,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nonlinear least-squares curve fitting via damped Gauss-Newton (Levenberg-Marquardt) on GPU."""
    xg = _to_f32(xdata).ravel()
    yg = _to_f32(ydata).ravel()
    params = np.asarray(p0, dtype=np.float64).copy()
    k = len(params)
    lam = 1e-3

    def _resid(p: np.ndarray) -> vnp.ndarray:
        return _to_f32(f(xg, *[float(v) for v in p])).ravel() - yg

    r = _resid(params)
    cost = float(vnp.sum(r * r).item())

    for _ in range(maxfev):
        # Build Jacobian columns (N, k) on GPU via finite differences
        J_cols = []
        for j in range(k):
            eps = max(1e-4, abs(params[j]) * 1e-4)
            p_plus = params.copy()
            p_plus[j] += eps
            rj = _resid(p_plus)
            J_cols.append(((rj - r) * (1.0 / eps)).reshape(-1, 1))
        J = vnp.concatenate(J_cols, axis=1)  # (N, k) on GPU

        # Normal equations on GPU -> tiny (k, k) float64 solve
        JtJ = (J.T @ J).to_numpy().astype(np.float64)
        Jtr = (J.T @ r.reshape(-1, 1)).to_numpy().astype(np.float64).ravel()

        A = JtJ + lam * np.diag(np.maximum(np.diag(JtJ), 1e-6))
        try:
            step = -np.linalg.solve(A, Jtr)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        cand = params + step
        r_cand = _resid(cand)
        cost_cand = float(vnp.sum(r_cand * r_cand).item())
        if cost_cand < cost:
            rel = abs(cost - cost_cand) / max(cost, 1e-12)
            params, r, cost = cand, r_cand, cost_cand
            lam = max(1e-7, lam * 0.3)
            if rel < tol:
                break
        else:
            lam = min(1e6, lam * 5.0)

    dof = max(1, xg.size - k)
    s_sq = cost / dof
    try:
        pcov = np.linalg.inv(JtJ) * s_sq
    except np.linalg.LinAlgError:
        pcov = np.full((k, k), np.inf, dtype=np.float64)
    return params, pcov


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: Any,
    lr: float = 0.05,
    maxiter: int = 300,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Adam-accelerated gradient-based minimization for scalar GPU/CPU objectives."""
    x = np.asarray(x0, dtype=np.float64).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    fval = float(fun(x))

    for it in range(1, maxiter + 1):
        grad = np.zeros_like(x)
        for j in range(len(x)):
            h = max(1e-5, abs(x[j]) * 1e-5)
            xp = x.copy(); xp[j] += h
            xm = x.copy(); xm[j] -= h
            grad[j] = (float(fun(xp)) - float(fun(xm))) / (2.0 * h)
        if np.linalg.norm(grad) < tol:
            return OptimizeResult(x=x, fun=fval, nit=it, success=True)
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * (grad * grad)
        mh = m / (1.0 - b1 ** it)
        vh = v / (1.0 - b2 ** it)
        x = x - lr * mh / (np.sqrt(vh) + eps_adam)
        fval = float(fun(x))

    return OptimizeResult(x=x, fun=fval, nit=maxiter, success=True)
