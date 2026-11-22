"""Vulkan GPU TriMap (Large-Scale Dimensionality Reduction Using Triplets)."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as _np
import vknumpy as vnp

from ._common import GPUAdam, init_embedding
from .knn import _to_f32_2d, knn_search


class TriMap:
    """Vulkan GPU TriMap embedding (``mlx-vis`` compatible API)."""

    def __init__(
        self,
        n_components: int = 2,
        n_inliers: int = 10,
        n_outliers: int = 4,
        n_random: int = 4,
        n_iters: int = 300,
        learning_rate: float = 0.5,
        metric: str = "euclidean",
        init: str = "pca",
        knn_method: str = "auto",
        seed: int = 42,
    ) -> None:
        self.n_components = int(n_components)
        self.n_inliers = int(n_inliers)
        self.n_outliers = int(n_outliers)
        self.n_random = int(n_random)
        self.n_iters = int(n_iters)
        self.learning_rate = float(learning_rate)
        self.metric = metric
        self.init = init
        self.knn_method = knn_method
        self.seed = int(seed)
        self.embedding_: Optional[_np.ndarray] = None
        self.history_: List[_np.ndarray] = []

    def fit_transform(
        self,
        X,
        *,
        record_every: int = 0,
        callback: Optional[Callable[[int, _np.ndarray], None]] = None,
    ) -> _np.ndarray:
        X_np = _to_f32_2d(X)
        n = X_np.shape[0]
        d_out = self.n_components
        k_in = min(self.n_inliers, max(2, n - 1))

        # 1. GPU K-NN for inlier triplets
        knn_idx, knn_dist = knn_search(X_np, k=k_in, metric=self.metric, method=self.knn_method, seed=self.seed)
        sigma = _np.maximum(_np.mean(knn_dist[:, :min(5, k_in)], axis=1), 1e-4)

        rng = _np.random.default_rng(self.seed + 404)
        n_out = max(1, self.n_outliers)
        t_per = k_in * n_out + self.n_random

        # Build per-anchor triplets (i, j_inlier, k_outlier) of shape (N, t_per)
        j_in = _np.repeat(knn_idx, n_out, axis=1)
        k_out = rng.integers(0, n, size=(n, k_in * n_out), dtype=_np.int32)

        # Tempered triplet weights from high-D scaled inlier distance
        d_in_s = _np.repeat(knn_dist / sigma[:, None], n_out, axis=1)
        w_in = _np.log1p(_np.exp(-d_in_s * 0.5) * 10.0).astype(_np.float32)

        if self.n_random > 0:
            r1 = rng.integers(0, n, size=(n, self.n_random), dtype=_np.int32)
            r2 = rng.integers(0, n, size=(n, self.n_random), dtype=_np.int32)
            j_all = _np.concatenate([j_in, r1], axis=1)
            k_all = _np.concatenate([k_out, r2], axis=1)
            w_all = _np.concatenate([w_in, _np.full((n, self.n_random), 0.25, dtype=_np.float32)], axis=1)
        else:
            j_all, k_all, w_all = j_in, k_out, w_in

        # 2. Initialize & upload to Vulkan GPU
        Y_init = init_embedding(X_np, n_components=d_out, init=self.init, scale=1.0, seed=self.seed)
        Y = vnp.array(Y_init, dtype=_np.float32)
        j_gpu = vnp.array(j_all.ravel(), dtype=_np.int32)
        k_gpu = vnp.array(k_all.ravel(), dtype=_np.int32)
        w_gpu = vnp.array(w_all, dtype=_np.float32)

        opt = GPUAdam((n, d_out), lr=self.learning_rate)

        self.history_ = []
        if record_every > 0:
            self.history_.append(Y_init.copy())

        for it in range(self.n_iters):
            Yj = vnp.take(Y, j_gpu, axis=0).reshape(n, t_per, d_out)
            Yk = vnp.take(Y, k_gpu, axis=0).reshape(n, t_per, d_out)
            Yi = Y.reshape(n, 1, d_out)

            diff_ij = Yi - Yj
            diff_ik = Yi - Yk
            d2_ij = vnp.sum(diff_ij * diff_ij, axis=2) + 1.0
            d2_ik = vnp.sum(diff_ik * diff_ik, axis=2) + 1.0

            # s_ij = 1/d2_ij, s_ik = 1/d2_ik; L = w * s_ik / (s_ij + s_ik) = w * d2_ij / (d2_ij + d2_ik)
            denom = d2_ij + d2_ik
            inv_denom2 = 1.0 / (denom * denom)
            c_ij = (2.0 * w_gpu * d2_ik) * inv_denom2
            c_ik = -(2.0 * w_gpu * d2_ij) * inv_denom2

            grad = vnp.sum(
                c_ij.reshape(n, t_per, 1) * diff_ij + c_ik.reshape(n, t_per, 1) * diff_ik,
                axis=1,
            ) * (1.0 / float(t_per))

            lr_scale = max(0.05, 1.0 - float(it) / float(max(1, self.n_iters)))
            Y = opt.step(Y, grad, lr_scale=lr_scale)

            if record_every > 0 and ((it + 1) % record_every == 0 or it == self.n_iters - 1):
                snap = Y.to_numpy()
                self.history_.append(snap.copy())
                if callback is not None:
                    callback(it + 1, snap)
            elif callback is not None and ((it + 1) % 50 == 0 or it == self.n_iters - 1):
                callback(it + 1, Y.to_numpy())

        self.embedding_ = Y.to_numpy().astype(_np.float32)
        return self.embedding_

    def fit(self, X) -> "TriMap":
        self.fit_transform(X)
        return self
