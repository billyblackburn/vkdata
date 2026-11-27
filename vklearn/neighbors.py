"""GPU NearestNeighbors and KNeighborsClassifier via ggml-vulkan GEMM + top_k."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_1d, to_gpu_2d
from .metrics import accuracy_score, r2_score


class NearestNeighbors:
    """Brute-force Euclidean k-nearest neighbors on Vulkan GPU (chunked GEMM + top_k)."""

    def __init__(self, n_neighbors: int = 5, chunk_size: int = 2048) -> None:
        self.n_neighbors = int(n_neighbors)
        self.chunk_size = int(chunk_size)
        self._X_fit: Optional[vnp.ndarray] = None
        self._x_norm2: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "NearestNeighbors":
        self._X_fit = to_gpu_2d(X)
        self._x_norm2 = vnp.sum(self._X_fit * self._X_fit, axis=1).reshape(1, -1)
        return self

    def kneighbors(
        self,
        X: Any,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
    ) -> Any:
        if self._X_fit is None or self._x_norm2 is None:
            raise RuntimeError("NearestNeighbors must be fitted before kneighbors()")
        k = self.n_neighbors if n_neighbors is None else int(n_neighbors)
        Qg = to_gpu_2d(X)
        m = Qg.shape[0]
        xt = self._X_fit.T

        idx_chunks = []
        dist_chunks = []
        for start in range(0, m, self.chunk_size):
            end = min(m, start + self.chunk_size)
            Qc = Qg[start:end]
            q_norm2 = vnp.sum(Qc * Qc, axis=1).reshape(-1, 1)
            sq_dists = vnp.maximum(q_norm2 + self._x_norm2 - 2.0 * (Qc @ xt), 0.0)
            # top_k on negative distances returns indices of k smallest distances!
            top_idx = vnp.top_k(-sq_dists, k=k, axis=1)
            idx_chunks.append(top_idx)
            if return_distance:
                d_np = sq_dists.to_numpy()
                i_np = top_idx.to_numpy()
                rows = np.arange(end - start)[:, None]
                d_sel = np.sqrt(np.maximum(d_np[rows, i_np], 0.0)).astype(np.float32)
                dist_chunks.append(vnp.array(d_sel, dtype=np.float32))

        indices = idx_chunks[0] if len(idx_chunks) == 1 else vnp.concatenate(idx_chunks, axis=0)
        if not return_distance:
            return indices
        distances = dist_chunks[0] if len(dist_chunks) == 1 else vnp.concatenate(dist_chunks, axis=0)
        return distances, indices


class KNeighborsClassifier(NearestNeighbors):
    """GPU k-Nearest Neighbors classifier using GEMM + top_k majority vote."""

    def __init__(self, n_neighbors: int = 5, chunk_size: int = 2048) -> None:
        super().__init__(n_neighbors=n_neighbors, chunk_size=chunk_size)
        self._y_fit: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "KNeighborsClassifier":
        super().fit(X)
        y_np = to_gpu_1d(y, np.int32).to_numpy()
        self.classes_, encoded = np.unique(y_np, return_inverse=True)
        self._y_fit = encoded.astype(np.int32)
        return self

    def predict(self, X: Any) -> vnp.ndarray:
        if self._y_fit is None or self.classes_ is None:
            raise RuntimeError("KNeighborsClassifier must be fitted before predict()")
        idx = self.kneighbors(X, return_distance=False).to_numpy()
        neigh_labels = self._y_fit[idx]  # (m, k)
        n_classes = len(self.classes_)
        # Fast vectorized majority vote across k neighbors
        votes = np.zeros((neigh_labels.shape[0], n_classes), dtype=np.int32)
        for c in range(n_classes):
            votes[:, c] = np.sum(neigh_labels == c, axis=1)
        pred_codes = np.argmax(votes, axis=1)
        return vnp.array(self.classes_[pred_codes].astype(np.int32), dtype=np.int32)

    def score(self, X: Any, y: Any) -> float:
        return accuracy_score(y, self.predict(X))


class KNeighborsRegressor(NearestNeighbors):
    """GPU k-Nearest Neighbors Regressor using ggml-vulkan GEMM distances."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        chunk_size: int = 2048,
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, chunk_size=chunk_size)
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
        self.weights = weights
        self._y_fit: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "KNeighborsRegressor":
        super().fit(X)
        self._y_fit = to_gpu_1d(y).to_numpy().astype(np.float32)
        return self

    def predict(self, X: Any) -> vnp.ndarray:
        if self._y_fit is None:
            raise RuntimeError("KNeighborsRegressor must be fitted before predict()")
        dists, idx = self.kneighbors(X, return_distance=True)
        idx_np = idx.to_numpy()
        neigh_y = self._y_fit[idx_np]  # (m, k)
        if self.weights == "uniform":
            preds = np.mean(neigh_y, axis=1).astype(np.float32)
        else:
            d_np = dists.to_numpy().astype(np.float64)
            w = 1.0 / np.maximum(d_np, 1e-8)
            preds = (np.sum(w * neigh_y, axis=1) / np.sum(w, axis=1)).astype(np.float32)
        return vnp.array(preds, dtype=np.float32)

    def score(self, X: Any, y: Any) -> float:
        return r2_score(y, self.predict(X))
