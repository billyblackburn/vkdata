"""Vulkan GPU PaCMAP & LocalMAP (Pairwise Controlled Manifold Approximation)."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as _np
import vknumpy as vnp

from ._common import GPUAdam, init_embedding
from .knn import _to_f32_2d, knn_search


class PaCMAP:
    """Vulkan GPU PaCMAP (3-phase Adam on Near, Mid-Near, and Further pairs)."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        MN_ratio: float = 0.5,
        FP_ratio: float = 2.0,
        n_iters: Tuple[int, int, int] = (100, 100, 200),
        learning_rate: float = 1.0,
        metric: str = "euclidean",
        init: str = "pca",
        knn_method: str = "auto",
        seed: int = 42,
    ) -> None:
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.MN_ratio = float(MN_ratio)
        self.FP_ratio = float(FP_ratio)
        self.n_iters = tuple(int(x) for x in n_iters)
        self.learning_rate = float(learning_rate)
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
        low_dist_thres: Optional[float] = None,
        record_every: int = 0,
        callback: Optional[Callable[[int, _np.ndarray], None]] = None,
    ) -> _np.ndarray:
        X_np = _to_f32_2d(X)
        n = X_np.shape[0]
        d_out = self.n_components
        k_nb = min(self.n_neighbors, max(2, n - 1))
        k_search = min(n - 1, max(k_nb + 10, 30))

        # 1. GPU K-NN & scaled-distance near-neighbor selection
        knn_idx, knn_dist = knn_search(X_np, k=k_search, metric=self.metric, method=self.knn_method, seed=self.seed)
        i_lo = min(3, k_search - 1)
        i_hi = min(6, k_search)
        sigma = _np.maximum(_np.mean(knn_dist[:, i_lo:i_hi], axis=1), 1e-4)
        scaled_d = (knn_dist * knn_dist) / _np.maximum(sigma[:, None] * sigma[knn_idx], 1e-8)
        ord_nb = _np.argsort(scaled_d, axis=1)[:, :k_nb]
        nb_idx = _np.take_along_axis(knn_idx, ord_nb, axis=1).astype(_np.int32)

        # 2. Sample Mid-Near (MN) pairs: for each slot, draw 6 random candidates and pick 2nd closest
        rng = _np.random.default_rng(self.seed + 303)
        k_mn = max(1, int(round(k_nb * self.MN_ratio)))
        mn_idx = _np.empty((n, k_mn), dtype=_np.int32)
        X_gpu = vnp.array(X_np, dtype=_np.float32)
        d_in = X_np.shape[1]
        for s in range(k_mn):
            cands = rng.integers(0, n, size=(n, 6), dtype=_np.int32)
            # Compute high-D distances to the 6 candidates on GPU
            Xn = vnp.take(X_gpu, vnp.array(cands.ravel(), dtype=_np.int32), axis=0).reshape(n, 6, d_in)
            diff = X_gpu.reshape(n, 1, d_in) - Xn
            d2 = vnp.sum(diff * diff, axis=2).to_numpy()
            second = _np.argsort(d2, axis=1)[:, 1]
            mn_idx[:, s] = cands[_np.arange(n), second]

        # 3. Sample Further Pairs (FP)
        k_fp = max(2, int(round(k_nb * self.FP_ratio)))
        fp_idx = rng.integers(0, n, size=(n, k_fp), dtype=_np.int32)

        # 4. Initialize embedding on GPU
        Y_init = init_embedding(X_np, n_components=d_out, init=self.init, scale=1.0, seed=self.seed)
        Y = vnp.array(Y_init, dtype=_np.float32)
        nb_gpu = vnp.array(nb_idx.ravel(), dtype=_np.int32)
        mn_gpu = vnp.array(mn_idx.ravel(), dtype=_np.int32)
        fp_gpu = vnp.array(fp_idx.ravel(), dtype=_np.int32)

        opt = GPUAdam((n, d_out), lr=self.learning_rate)
        n1, n2, n3 = self.n_iters
        total_iters = n1 + n2 + n3

        self.history_ = []
        if record_every > 0:
            self.history_.append(Y_init.copy())

        for it in range(total_iters):
            if it < n1:
                w_nb = 2.0
                w_mn = 1000.0 * (1.0 - it / max(1, n1)) + 3.0 * (it / max(1, n1))
                w_fp = 1.0
            elif it < n1 + n2:
                w_nb = 3.0
                w_mn = 3.0
                w_fp = 1.0
            else:
                w_nb = 1.0 if low_dist_thres is None else 1.5
                w_mn = 0.0
                w_fp = 1.0

            # Near-neighbor attractive gradient
            Y_nb = vnp.take(Y, nb_gpu, axis=0).reshape(n, k_nb, d_out)
            diff_nb = Y.reshape(n, 1, d_out) - Y_nb
            d_tilde_nb = vnp.sum(diff_nb * diff_nb, axis=2) + 1.0
            denom_nb = 10.0 + d_tilde_nb
            c_nb = (w_nb * 20.0) / (denom_nb * denom_nb)
            grad = vnp.sum(c_nb.reshape(n, k_nb, 1) * diff_nb, axis=1)

            # Mid-near attractive gradient (phases 1 & 2)
            if w_mn > 0.0:
                Y_mn = vnp.take(Y, mn_gpu, axis=0).reshape(n, k_mn, d_out)
                diff_mn = Y.reshape(n, 1, d_out) - Y_mn
                d_tilde_mn = vnp.sum(diff_mn * diff_mn, axis=2) + 1.0
                denom_mn = 10000.0 + d_tilde_mn
                c_mn = (w_mn * 20000.0) / (denom_mn * denom_mn)
                grad = grad + vnp.sum(c_mn.reshape(n, k_mn, 1) * diff_mn, axis=1)

            # Further-pair repulsive gradient
            Y_fp = vnp.take(Y, fp_gpu, axis=0).reshape(n, k_fp, d_out)
            diff_fp = Y.reshape(n, 1, d_out) - Y_fp
            d_tilde_fp = vnp.sum(diff_fp * diff_fp, axis=2) + 1.0
            denom_fp = 1.0 + d_tilde_fp
            c_fp = -(w_fp * 2.0) / (denom_fp * denom_fp)

            # LocalMAP: in Phase 3, focus further-pair repulsion on locally close non-neighbors
            if low_dist_thres is not None and it >= n1 + n2:
                thres_sq = float(low_dist_thres * low_dist_thres) + 1.0
                local_mask = vnp.where(d_tilde_fp < thres_sq, 1.5, 0.25)
                c_fp = c_fp * local_mask

            grad = grad + vnp.sum(c_fp.reshape(n, k_fp, 1) * diff_fp, axis=1)
            Y = opt.step(Y, grad)

            if record_every > 0 and ((it + 1) % record_every == 0 or it == total_iters - 1):
                snap = Y.to_numpy()
                self.history_.append(snap.copy())
                if callback is not None:
                    callback(it + 1, snap)
            elif callback is not None and ((it + 1) % 50 == 0 or it == total_iters - 1):
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
        return self._fit_core(X, low_dist_thres=None, record_every=record_every, callback=callback)

    def fit(self, X) -> "PaCMAP":
        self.fit_transform(X)
        return self


class LocalMAP(PaCMAP):
    """LocalMAP: PaCMAP variant with dynamic local-graph adjustment in Phase 3."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        low_dist_thres: float = 10.0,
        MN_ratio: float = 0.5,
        FP_ratio: float = 2.0,
        n_iters: Tuple[int, int, int] = (100, 100, 200),
        learning_rate: float = 1.0,
        metric: str = "euclidean",
        init: str = "pca",
        knn_method: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            n_iters=n_iters,
            learning_rate=learning_rate,
            metric=metric,
            init=init,
            knn_method=knn_method,
            seed=seed,
        )
        self.low_dist_thres = float(low_dist_thres)

    def fit_transform(
        self,
        X,
        *,
        record_every: int = 0,
        callback: Optional[Callable[[int, _np.ndarray], None]] = None,
    ) -> _np.ndarray:
        return self._fit_core(X, low_dist_thres=self.low_dist_thres, record_every=record_every, callback=callback)
