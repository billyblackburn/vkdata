"""Vulkan GPU UMAP (Uniform Manifold Approximation and Projection)."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as _np
import vknumpy as vnp

from ._common import GPUAdam, fit_umap_ab, init_embedding, umap_fuzzy_weights
from .knn import _to_f32_2d, knn_search, symmetrize_knn_table


class UMAP:
    """Vulkan GPU UMAP embedding (``mlx-vis`` compatible API).

    Parameters
    ----------
    n_components : int, default=2
        Target embedding dimension (2 or 3).
    n_neighbors : int, default=15
        Size of local neighborhood for fuzzy simplicial set construction.
    min_dist : float, default=0.1
        Minimum distance between embedded points.
    spread : float, default=1.0
        Effective scale of embedded points.
    n_epochs : int, default=200
        Number of GPU optimization epochs.
    learning_rate : float, default=1.0
        Initial Adam learning rate.
    negative_sample_rate : int, default=5
        Number of repulsive negative samples per positive neighbor.
    metric : str, default="euclidean"
        Distance metric (``"euclidean"`` or ``"cosine"``).
    init : str or ndarray, default="pca"
        Initialization method (``"pca"``, ``"random"``, or initial coordinates).
    knn_method : str, default="auto"
        ``"auto"``, ``"brute"``, or ``"nndescent"``.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        metric: str = "euclidean",
        init: str = "pca",
        knn_method: str = "auto",
        seed: int = 42,
    ) -> None:
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.spread = float(spread)
        self.n_epochs = int(n_epochs)
        self.learning_rate = float(learning_rate)
        self.negative_sample_rate = int(negative_sample_rate)
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
        """Fit UMAP on Vulkan GPU and return the ``(N, n_components)`` float32 embedding."""
        X_np = _to_f32_2d(X)
        n = X_np.shape[0]
        d_out = self.n_components
        k = min(self.n_neighbors, max(2, n - 1))

        # 1. GPU K-NN graph & fuzzy simplicial set
        knn_idx, knn_dist = knn_search(X_np, k=k, metric=self.metric, method=self.knn_method, seed=self.seed)
        w_dir = umap_fuzzy_weights(knn_dist, k=k)
        k_sym = min(n - 1, max(k, int(k * 1.6)))
        sym_idx, sym_w = symmetrize_knn_table(knn_idx, w_dir, max_deg=k_sym, mode="fuzzy")

        # 2. Curve parameters (a, b) and initialization
        a, b = fit_umap_ab(self.spread, self.min_dist)
        Y_np = init_embedding(X_np, n_components=d_out, init=self.init, scale=8.0, seed=self.seed)

        # 3. Upload fixed graph & initial embedding to Vulkan GPU
        Y = vnp.array(Y_np, dtype=_np.float32)
        nbr_idx_gpu = vnp.array(sym_idx.ravel(), dtype=_np.int32)
        nbr_w_gpu = vnp.array(sym_w, dtype=_np.float32)

        m_neg = min(n - 1, max(5, int(self.negative_sample_rate * 2)))
        opt = GPUAdam((n, d_out), lr=self.learning_rate)
        rng = _np.random.default_rng(self.seed + 101)

        self.history_ = []
        if record_every > 0:
            self.history_.append(Y_np.copy())

        two_ab = float(2.0 * a * b)
        two_b = float(2.0 * b)
        bm1 = float(b - 1.0)
        b_f = float(b)
        a_f = float(a)
        rep_scale = float(k) / float(m_neg)

        for epoch in range(self.n_epochs):
            # Gather positive neighbors: (N, k_sym, d_out)
            Y_pos = vnp.take(Y, nbr_idx_gpu, axis=0).reshape(n, k_sym, d_out)
            diff_pos = Y.reshape(n, 1, d_out) - Y_pos
            d2_pos = vnp.sum(diff_pos * diff_pos, axis=2) + 1e-4

            # Attractive gradient: w * 2ab * d2^(b-1) / (1 + a * d2^b) * diff
            log_d2 = vnp.log(d2_pos)
            d2_b = vnp.exp(log_d2 * b_f)
            d2_bm1 = vnp.exp(log_d2 * bm1)
            coeff_pos = nbr_w_gpu * (two_ab * d2_bm1 / (1.0 + a_f * d2_b))
            grad_pos = vnp.sum(vnp.clip(coeff_pos.reshape(n, k_sym, 1) * diff_pos, -4.0, 4.0), axis=1)

            # Negative sampling on GPU
            neg_idx_np = rng.integers(0, n, size=(n * m_neg,), dtype=_np.int32)
            neg_idx_gpu = vnp.array(neg_idx_np, dtype=_np.int32)
            Y_neg = vnp.take(Y, neg_idx_gpu, axis=0).reshape(n, m_neg, d_out)
            diff_neg = Y.reshape(n, 1, d_out) - Y_neg
            d2_neg = vnp.sum(diff_neg * diff_neg, axis=2) + 1e-3

            d2_neg_b = vnp.exp(vnp.log(d2_neg) * b_f)
            coeff_neg = -(two_b * rep_scale) / (d2_neg * (1.0 + a_f * d2_neg_b))
            grad_neg = vnp.sum(vnp.clip(coeff_neg.reshape(n, m_neg, 1) * diff_neg, -4.0, 4.0), axis=1)

            grad = (grad_pos + grad_neg) * (1.0 / max(1.0, float(k)))
            lr_scale = max(0.05, 1.0 - float(epoch) / float(max(1, self.n_epochs)))
            Y = opt.step(Y, grad, lr_scale=lr_scale)

            if record_every > 0 and ((epoch + 1) % record_every == 0 or epoch == self.n_epochs - 1):
                snap = Y.to_numpy()
                self.history_.append(snap.copy())
                if callback is not None:
                    callback(epoch + 1, snap)
            elif callback is not None and ((epoch + 1) % 25 == 0 or epoch == self.n_epochs - 1):
                callback(epoch + 1, Y.to_numpy())

        self.embedding_ = Y.to_numpy().astype(_np.float32)
        return self.embedding_

    def fit(self, X) -> "UMAP":
        self.fit_transform(X)
        return self
