"""Vulkan GPU t-SNE and DREAMS (t-SNE + PCA global-structure regularization)."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as _np
import vknumpy as vnp

from ._common import entropic_affinities, init_embedding
from .knn import _to_f32_2d, knn_search, symmetrize_knn_table


class TSNE:
    """Vulkan GPU t-Distributed Stochastic Neighbor Embedding (``mlx-vis`` API).

    Uses exact GPU GEMM Student-t repulsion for ``N <= 4096`` and negative-sampled
    Student-t repulsion for larger datasets, with early exaggeration and momentum.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iters: int = 400,
        learning_rate: float = 200.0,
        early_exaggeration: float = 12.0,
        metric: str = "euclidean",
        init: str = "pca",
        knn_method: str = "auto",
        seed: int = 42,
    ) -> None:
        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.n_iters = int(n_iters)
        self.learning_rate = float(learning_rate)
        self.early_exaggeration = float(early_exaggeration)
        self.metric = metric
        self.init = init
        self.knn_method = knn_method
        self.seed = int(seed)
        self.embedding_: Optional[_np.ndarray] = None
        self.history_: List[_np.ndarray] = []

    def _fit_core(
        self,
        X,
        *,
        pca_lam: float = 0.0,
        record_every: int = 0,
        callback: Optional[Callable[[int, _np.ndarray], None]] = None,
    ) -> _np.ndarray:
        X_np = _to_f32_2d(X)
        n = X_np.shape[0]
        d_out = self.n_components
        perp = min(self.perplexity, max(2.0, (n - 1) / 3.0))
        k = min(n - 1, max(8, int(round(3.0 * perp))))

        # 1. GPU K-NN & entropic affinities P_{j|i}
        knn_idx, knn_dist = knn_search(X_np, k=k, metric=self.metric, method=self.knn_method, seed=self.seed)
        p_cond = entropic_affinities(knn_dist, perplexity=perp)
        k_sym = min(n - 1, max(k, int(k * 1.5)))
        sym_idx, sym_p = symmetrize_knn_table(knn_idx, p_cond, max_deg=k_sym, mode="mean")
        # Normalize P so sum(P) = 1.0
        p_sum = float(_np.sum(sym_p)) + 1e-12
        sym_p = (sym_p / p_sum).astype(_np.float32)

        # 2. Initialize Y (small scale for classic t-SNE)
        Y_init = init_embedding(X_np, n_components=d_out, init=self.init, scale=1.0, seed=self.seed)
        Y = vnp.array(Y_init, dtype=_np.float32)
        Y_pca = vnp.array(
            init_embedding(X_np, n_components=d_out, init="pca", scale=15.0, seed=self.seed),
            dtype=_np.float32,
        ) if pca_lam > 0.0 else None

        nbr_idx_gpu = vnp.array(sym_idx.ravel(), dtype=_np.int32)
        nbr_p_gpu = vnp.array(sym_p, dtype=_np.float32)

        vel = vnp.zeros((n, d_out), dtype=_np.float32)
        gains = vnp.ones((n, d_out), dtype=_np.float32)
        prev_grad = vnp.zeros((n, d_out), dtype=_np.float32)

        use_exact_rep = n <= 2048
        eye_mask = vnp.array(1.0 - _np.eye(n, dtype=_np.float32), dtype=_np.float32) if use_exact_rep else None
        rng = _np.random.default_rng(self.seed + 202)
        m_neg = min(n - 1, 48)
        exag_stop = max(50, self.n_iters // 4)

        self.history_ = []
        if record_every > 0:
            self.history_.append(Y_init.copy())

        for it in range(self.n_iters):
            exag = self.early_exaggeration if it < exag_stop else 1.0
            momentum = 0.5 if it < exag_stop else 0.8

            # Attractive force on sparse symmetric neighbor graph
            Y_pos = vnp.take(Y, nbr_idx_gpu, axis=0).reshape(n, k_sym, d_out)
            diff_pos = Y.reshape(n, 1, d_out) - Y_pos
            d2_pos = vnp.sum(diff_pos * diff_pos, axis=2)
            q_pos = 1.0 / (1.0 + d2_pos)
            grad_attr = vnp.sum((nbr_p_gpu * q_pos * (4.0 * exag)).reshape(n, k_sym, 1) * diff_pos, axis=1)

            # Repulsive force
            if use_exact_rep:
                # Exact O(N^2) Student-t repulsion via a single GPU GEMM!
                sq = vnp.sum(Y * Y, axis=1)
                d2_all = vnp.maximum(sq.reshape(n, 1) + sq.reshape(1, n) - (Y @ Y.T.contiguous()) * 2.0, 0.0)
                q_num = (1.0 / (1.0 + d2_all)) * eye_mask
                Z = vnp.maximum(vnp.sum(q_num), 1e-12)
                W = (q_num * q_num) / Z
                row_w = vnp.sum(W, axis=1).reshape(n, 1)
                grad_rep = (row_w * Y - (W @ Y)) * (-4.0)
            else:
                # Negative-sampled repulsion for large N > 4096
                neg_idx = vnp.array(rng.integers(0, n, size=(n * m_neg,), dtype=_np.int32), dtype=_np.int32)
                Y_neg = vnp.take(Y, neg_idx, axis=0).reshape(n, m_neg, d_out)
                diff_neg = Y.reshape(n, 1, d_out) - Y_neg
                d2_neg = vnp.sum(diff_neg * diff_neg, axis=2)
                q_neg = 1.0 / (1.0 + d2_neg)
                z_est = vnp.maximum(vnp.sum(q_neg), 1e-8)
                coeff_neg = -(4.0 / z_est) * (q_neg * q_neg)
                grad_rep = vnp.sum(coeff_neg.reshape(n, m_neg, 1) * diff_neg, axis=1)

            grad = grad_attr + grad_rep
            if Y_pca is not None and it >= exag_stop:
                # DREAMS global PCA regularization term
                grad = grad + (Y - Y_pca) * (2.0 * pca_lam / float(n))

            # Jacobs adaptive gains + momentum on GPU
            if it > 0:
                same_sign = (grad * prev_grad) > 0.0
                gains = vnp.maximum(vnp.where(same_sign, gains * 0.8, gains + 0.2), 0.01)
            prev_grad = grad

            vel = vel * momentum - (gains * grad) * self.learning_rate
            Y = Y + vel
            # Re-center embedding to zero mean
            Y = Y - vnp.mean(Y, axis=0).reshape(1, d_out)

            if record_every > 0 and ((it + 1) % record_every == 0 or it == self.n_iters - 1):
                snap = Y.to_numpy()
                self.history_.append(snap.copy())
                if callback is not None:
                    callback(it + 1, snap)
            elif callback is not None and ((it + 1) % 50 == 0 or it == self.n_iters - 1):
                callback(it + 1, Y.to_numpy())

        self.embedding_ = Y.to_numpy().astype(_np.float32)
        return self.embedding_

    def fit_transform(
        self,
        X,
        *,
        record_every: int = 0,
        callback: Optional[Callable[[int, _np.ndarray], None]] = None,
    ) -> _np.ndarray:
        return self._fit_core(X, pca_lam=0.0, record_every=record_every, callback=callback)

    def fit(self, X) -> "TSNE":
        self.fit_transform(X)
        return self


class DREAMS(TSNE):
    """DREAMS: t-SNE + PCA global structure regularization hybrid (``mlx-vis`` API)."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        lam: float = 0.15,
        n_iters: int = 400,
        learning_rate: float = 200.0,
        early_exaggeration: float = 12.0,
        metric: str = "euclidean",
        init: str = "pca",
        knn_method: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_components=n_components,
            perplexity=perplexity,
            n_iters=n_iters,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            metric=metric,
            init=init,
            knn_method=knn_method,
            seed=seed,
        )
        self.lam = float(lam)

    def fit_transform(
        self,
        X,
        *,
        record_every: int = 0,
        callback: Optional[Callable[[int, _np.ndarray], None]] = None,
    ) -> _np.ndarray:
        return self._fit_core(X, pca_lam=self.lam, record_every=record_every, callback=callback)
