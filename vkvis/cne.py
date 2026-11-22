"""Vulkan GPU CNE (Contrastive Neighbor Embedding: InfoNCE / NCE / Neg-Sampling)."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as _np
import vknumpy as vnp

from ._common import GPUAdam, init_embedding, umap_fuzzy_weights
from .knn import _to_f32_2d, knn_search, symmetrize_knn_table


class CNE:
    """Contrastive Neighbor Embedding unifying t-SNE (``infonce``) and UMAP (``neg``/``nce``)."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        loss: str = "infonce",
        negative_sample_rate: int = 5,
        temperature: float = 0.5,
        n_epochs: int = 220,
        learning_rate: float = 0.8,
        metric: str = "euclidean",
        init: str = "pca",
        knn_method: str = "auto",
        seed: int = 42,
    ) -> None:
        if loss not in ("infonce", "nce", "neg"):
            raise ValueError(f"Unsupported loss='{loss}'. Use 'infonce', 'nce', or 'neg'.")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.loss = loss
        self.negative_sample_rate = int(negative_sample_rate)
        self.temperature = float(temperature)
        self.n_epochs = int(n_epochs)
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
        k = min(self.n_neighbors, max(2, n - 1))

        knn_idx, knn_dist = knn_search(X_np, k=k, metric=self.metric, method=self.knn_method, seed=self.seed)
        w_dir = umap_fuzzy_weights(knn_dist, k=k)
        k_sym = min(n - 1, max(k, int(k * 1.5)))
        sym_idx, sym_w = symmetrize_knn_table(knn_idx, w_dir, max_deg=k_sym, mode="fuzzy")

        Y_init = init_embedding(X_np, n_components=d_out, init=self.init, scale=4.0, seed=self.seed)
        Y = vnp.array(Y_init, dtype=_np.float32)
        nbr_idx_gpu = vnp.array(sym_idx.ravel(), dtype=_np.int32)
        nbr_w_gpu = vnp.array(sym_w, dtype=_np.float32)

        m_neg = min(n - 1, max(8, self.negative_sample_rate * 3))
        opt = GPUAdam((n, d_out), lr=self.learning_rate)
        rng = _np.random.default_rng(self.seed + 505)
        inv_tau = 1.0 / max(self.temperature, 1e-3)

        self.history_ = []
        if record_every > 0:
            self.history_.append(Y_init.copy())

        for epoch in range(self.n_epochs):
            # Positive Cauchy similarities q_ij = 1 / (1 + ||y_i - y_j||^2 / tau)
            Y_pos = vnp.take(Y, nbr_idx_gpu, axis=0).reshape(n, k_sym, d_out)
            diff_pos = Y.reshape(n, 1, d_out) - Y_pos
            d2_pos = vnp.sum(diff_pos * diff_pos, axis=2)
            q_pos = 1.0 / (1.0 + d2_pos * inv_tau)

            # Negative Cauchy similarities
            neg_idx = vnp.array(rng.integers(0, n, size=(n * m_neg,), dtype=_np.int32), dtype=_np.int32)
            Y_neg = vnp.take(Y, neg_idx, axis=0).reshape(n, m_neg, d_out)
            diff_neg = Y.reshape(n, 1, d_out) - Y_neg
            d2_neg = vnp.sum(diff_neg * diff_neg, axis=2)
            q_neg = 1.0 / (1.0 + d2_neg * inv_tau)

            if self.loss == "infonce":
                # InfoNCE: -log(q_pos / (q_pos + sum(q_neg)))
                z_neg = vnp.sum(q_neg, axis=1).reshape(n, 1) + 1e-6
                # Attractive weight: w * (z_neg / (q_pos + z_neg)) * (2/tau) * q_pos
                c_pos = nbr_w_gpu * (z_neg / (q_pos + z_neg)) * (2.0 * inv_tau) * q_pos
                # Repulsive weight: - (sum_w_i * q_neg / (mean_q_pos + z_neg)) * (2/tau) * q_neg
                w_row = vnp.sum(nbr_w_gpu, axis=1).reshape(n, 1)
                c_neg = -(w_row / z_neg) * (2.0 * inv_tau) * (q_neg * q_neg)
            elif self.loss == "nce":
                # NCE: -log(q_pos / (q_pos + Z)) - log(Z / (q_neg + Z))
                Z = float(m_neg) / float(max(1, n))
                c_pos = nbr_w_gpu * (Z / (q_pos + Z)) * (2.0 * inv_tau) * q_pos
                c_neg = -(float(k) / float(m_neg)) * (q_neg / (q_neg + Z)) * (2.0 * inv_tau) * q_neg
            else:
                # Negative sampling ("neg"): -log(q_pos) - log(1 - q_neg)
                c_pos = nbr_w_gpu * (2.0 * inv_tau) * q_pos
                c_neg = -(float(k) / float(m_neg)) * (2.0 * inv_tau) * (q_neg * q_neg) / vnp.maximum(1.0 - q_neg, 1e-3)

            grad_pos = vnp.sum(vnp.clip(c_pos.reshape(n, k_sym, 1) * diff_pos, -4.0, 4.0), axis=1)
            grad_neg = vnp.sum(vnp.clip(c_neg.reshape(n, m_neg, 1) * diff_neg, -4.0, 4.0), axis=1)
            grad = (grad_pos + grad_neg) * (1.0 / float(max(1, k)))

            lr_scale = max(0.05, 1.0 - float(epoch) / float(max(1, self.n_epochs)))
            Y = opt.step(Y, grad, lr_scale=lr_scale)

            if record_every > 0 and ((epoch + 1) % record_every == 0 or epoch == self.n_epochs - 1):
                snap = Y.to_numpy()
                self.history_.append(snap.copy())
                if callback is not None:
                    callback(epoch + 1, snap)
            elif callback is not None and ((epoch + 1) % 50 == 0 or epoch == self.n_epochs - 1):
                callback(epoch + 1, Y.to_numpy())

        self.embedding_ = Y.to_numpy().astype(_np.float32)
        return self.embedding_

    def fit(self, X) -> "CNE":
        self.fit_transform(X)
        return self
