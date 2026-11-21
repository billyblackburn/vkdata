"""GPU linear models: LinearRegression, Ridge, and LogisticRegression (IRLS / Newton)."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_1d, to_gpu_2d
from .metrics import accuracy_score, r2_score


class Ridge:
    """L2-regularized linear regression solved via GPU Gram GEMM + float64 Cholesky/LU."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.coef_: Optional[vnp.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: Any, y: Any) -> "Ridge":
        Xg = to_gpu_2d(X)
        yg = to_gpu_1d(y, np.float32)
        n, d = Xg.shape

        if self.fit_intercept:
            x_mean = vnp.mean(Xg, axis=0)
            y_mean = float(vnp.mean(yg).item())
            Xc = Xg - x_mean
            yc = yg - y_mean
        else:
            x_mean = None
            y_mean = 0.0
            Xc = Xg
            yc = yg

        # O(N * D^2) Gram matrix and O(N * D) RHS on ggml-vulkan GEMM
        G = (Xc.T @ Xc).to_numpy().astype(np.float64)
        rhs = (Xc.T @ yc.reshape(n, 1)).to_numpy().ravel().astype(np.float64)

        if self.alpha > 0.0:
            G[np.arange(d), np.arange(d)] += self.alpha

        try:
            w = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(G, rhs, rcond=None)[0]

        self.coef_ = vnp.array(w.astype(np.float32), dtype=np.float32)
        if self.fit_intercept and x_mean is not None:
            self.intercept_ = float(y_mean - float(np.dot(x_mean.to_numpy().astype(np.float64), w)))
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: Any) -> vnp.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict()")
        Xg = to_gpu_2d(X)
        pred = (Xg @ self.coef_.reshape(-1, 1)).ravel()
        return pred + self.intercept_ if self.fit_intercept else pred

    def score(self, X: Any, y: Any) -> float:
        return r2_score(y, self.predict(X))


class LinearRegression(Ridge):
    """Ordinary least squares linear regression on Vulkan GPU."""

    def __init__(self, fit_intercept: bool = True) -> None:
        super().__init__(alpha=0.0, fit_intercept=fit_intercept)


class LogisticRegression:
    """Binary L2-regularized Logistic Regression using GPU Newton-IRLS (ggml-vulkan GEMM)."""

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 25,
        tol: float = 1e-4,
    ) -> None:
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: Optional[vnp.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: Any, y: Any) -> "LogisticRegression":
        Xg = to_gpu_2d(X)
        yg = to_gpu_1d(y, np.float32)
        n, d = Xg.shape

        if self.fit_intercept:
            ones = vnp.ones((n, 1), dtype=np.float32)
            Xa = vnp.hstack([Xg, ones])
            p_dim = d + 1
        else:
            Xa = Xg
            p_dim = d

        lam = (1.0 / max(self.C, 1e-12))
        w = np.zeros((p_dim,), dtype=np.float64)
        reg = np.full((p_dim,), lam, dtype=np.float64)
        if self.fit_intercept:
            reg[-1] = 0.0  # Do not penalize intercept

        for it in range(1, self.max_iter + 1):
            w_gpu = vnp.array(w.astype(np.float32), dtype=np.float32).reshape(p_dim, 1)
            z = (Xa @ w_gpu).ravel()
            p = vnp.sigmoid(z)

            # Gradient: X^T (p - y) + reg * w
            diff = (p - yg).reshape(n, 1)
            grad = (Xa.T @ diff).to_numpy().ravel().astype(np.float64) + reg * w

            # Hessian: X^T diag(p*(1-p)) X + diag(reg) via GPU GEMM
            s = vnp.clip(p * (1.0 - p), 1e-6, 0.25).reshape(n, 1)
            Xw = Xa * s
            H = (Xa.T @ Xw).to_numpy().astype(np.float64)
            H[np.arange(p_dim), np.arange(p_dim)] += reg + 1e-6

            step = np.linalg.solve(H, grad)
            w -= step
            self.n_iter_ = it
            if float(np.max(np.abs(step))) < self.tol:
                break

        if self.fit_intercept:
            self.coef_ = vnp.array(w[:-1].astype(np.float32), dtype=np.float32)
            self.intercept_ = float(w[-1])
        else:
            self.coef_ = vnp.array(w.astype(np.float32), dtype=np.float32)
            self.intercept_ = 0.0
        return self

    def predict_proba(self, X: Any) -> vnp.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticRegression must be fitted before predict_proba()")
        Xg = to_gpu_2d(X)
        z = (Xg @ self.coef_.reshape(-1, 1)).ravel() + self.intercept_
        p1 = vnp.sigmoid(z)
        p0 = 1.0 - p1
        return vnp.stack([p0, p1], axis=1)

    def predict(self, X: Any) -> vnp.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticRegression must be fitted before predict()")
        Xg = to_gpu_2d(X)
        z = (Xg @ self.coef_.reshape(-1, 1)).ravel() + self.intercept_
        return (z >= 0.0).astype(np.int32)

    def score(self, X: Any, y: Any) -> float:
        return accuracy_score(y, self.predict(X))
