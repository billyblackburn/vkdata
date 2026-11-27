"""GPU K-Means clustering (k-means++ + chunked GEMM distances + O(N) GPU bincount centroids)."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_2d


class KMeans:
    """K-Means clustering on Vulkan GPU with k-means++ initialization and O(N) atomicAdd updates."""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        chunk_size: int = 524_288,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.chunk_size = int(chunk_size)
        self.cluster_centers_: Optional[vnp.ndarray] = None
        self._shift_mean: Optional[vnp.ndarray] = None
        self.labels_: Optional[vnp.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0

    def _init_kmeans_pp(self, Xg: vnp.ndarray) -> vnp.ndarray:
        """k-means++ seeding (uses a random GPU sample when N is very large)."""
        n, d = Xg.shape
        k = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        if n > 20_000:
            samp_idx = vnp.array(rng.choice(n, size=20_000, replace=False).astype(np.int32), dtype=np.int32)
            X_s = vnp.take(Xg, samp_idx, axis=0).to_numpy().astype(np.float64)
        else:
            X_s = Xg.to_numpy().astype(np.float64)

        m = X_s.shape[0]
        centers = np.empty((k, d), dtype=np.float64)
        centers[0] = X_s[int(rng.integers(0, m))]
        min_sq = np.sum((X_s - centers[0]) ** 2, axis=1)

        for i in range(1, k):
            total = float(min_sq.sum())
            if total <= 0.0:
                centers[i] = X_s[int(rng.integers(0, m))]
            else:
                probs = min_sq / total
                idx = int(rng.choice(m, p=probs))
                centers[i] = X_s[idx]
            sq = np.sum((X_s - centers[i]) ** 2, axis=1)
            min_sq = np.minimum(min_sq, sq)

        return vnp.array(centers.astype(np.float32), dtype=np.float32)

    def _assign_labels(
        self,
        Xg: vnp.ndarray,
        centers: vnp.ndarray,
        compute_inertia: bool = False,
    ) -> Tuple[vnp.ndarray, float]:
        """Chunked GEMM Euclidean distance assignment: ||x||^2 + ||c||^2 - 2 X C^T."""
        n = Xg.shape[0]
        c_norm2 = vnp.sum(centers * centers, axis=1).reshape(1, -1)
        c_t = centers.T

        label_chunks = []
        inertia = 0.0

        for start in range(0, n, self.chunk_size):
            end = min(n, start + self.chunk_size)
            Xc = Xg[start:end]
            x_norm2 = vnp.sum(Xc * Xc, axis=1).reshape(-1, 1)
            # Squared distances (chunk, K) on ggml-vulkan GEMM
            dists = vnp.maximum(x_norm2 + c_norm2 - 2.0 * (Xc @ c_t), 0.0)
            lbl = vnp.argmin(dists, axis=1)
            label_chunks.append(lbl)
            if compute_inertia:
                inertia += float(vnp.sum(vnp.min(dists, axis=1)).item())

        labels = label_chunks[0] if len(label_chunks) == 1 else vnp.concatenate(label_chunks, axis=0)
        return labels, inertia

    def fit(self, X: Any, y: Any = None) -> "KMeans":
        Xg = to_gpu_2d(X)
        n, d = Xg.shape
        k = self.n_clusters
        if k <= 0 or k > n:
            raise ValueError(f"n_clusters={k} must be in [1, {n}]")

        # Center data to prevent float32 catastrophic cancellation in ||x||^2 + ||c||^2 - 2 x.c
        self._shift_mean = vnp.mean(Xg, axis=0)
        Xc = Xg - self._shift_mean

        centers = self._init_kmeans_pp(Xc)
        labels = vnp.zeros((n,), dtype=np.int32)
        cols = [Xc[:, j].contiguous() for j in range(d)]

        for it in range(1, self.max_iter + 1):
            labels, _ = self._assign_labels(Xc, centers, compute_inertia=False)

            counts_np = vnp.bincount(labels, minlength=k).to_numpy().astype(np.float64)
            old_centers_np = centers.to_numpy().astype(np.float64)

            new_cols = []
            for j in range(d):
                sums = vnp.bincount(labels, weights=cols[j], minlength=k).to_numpy().astype(np.float64)
                mean_j = np.where(counts_np > 0, sums / np.maximum(counts_np, 1.0), old_centers_np[:, j])
                new_cols.append(mean_j.astype(np.float32))

            new_centers_np = np.stack(new_cols, axis=1)
            shift = float(np.sqrt(np.sum((new_centers_np - old_centers_np) ** 2)))
            centers = vnp.array(new_centers_np, dtype=np.float32)
            self.n_iter_ = it
            if shift <= self.tol:
                break

        labels, inertia = self._assign_labels(Xc, centers, compute_inertia=True)
        self.cluster_centers_ = centers + self._shift_mean
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, X: Any) -> vnp.ndarray:
        if self.cluster_centers_ is None or self._shift_mean is None:
            raise RuntimeError("KMeans must be fitted before predict()")
        Xc = to_gpu_2d(X) - self._shift_mean
        centers_c = self.cluster_centers_ - self._shift_mean
        labels, _ = self._assign_labels(Xc, centers_c, compute_inertia=False)
        return labels

    def fit_predict(self, X: Any, y: Any = None) -> vnp.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_


class DBSCAN:
    """Density-Based Spatial Clustering (DBSCAN) with Vulkan GPU GEMM pairwise distances."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        chunk_size: int = 2048,
    ) -> None:
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.chunk_size = int(chunk_size)
        self.labels_: Optional[vnp.ndarray] = None
        self.core_sample_indices_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "DBSCAN":
        Xg = to_gpu_2d(X)
        n = Xg.shape[0]
        eps2 = self.eps * self.eps

        Xc = Xg - vnp.mean(Xg, axis=0)
        x_norm2 = vnp.sum(Xc * Xc, axis=1)
        x_norm2_row = x_norm2.reshape(1, -1)
        xt = Xc.T

        neighbors: list[np.ndarray] = []
        core_list: list[int] = []

        for start in range(0, n, self.chunk_size):
            end = min(n, start + self.chunk_size)
            Xb = Xc[start:end]
            b_norm2 = x_norm2[start:end].reshape(-1, 1)
            d2 = vnp.maximum(b_norm2 + x_norm2_row - 2.0 * (Xb @ xt), 0.0).to_numpy()
            for local_i in range(end - start):
                nbrs = np.flatnonzero(d2[local_i] <= eps2)
                neighbors.append(nbrs)
                if len(nbrs) >= self.min_samples:
                    core_list.append(start + local_i)

        is_core = np.zeros(n, dtype=bool)
        if core_list:
            is_core[core_list] = True

        labels = np.full(n, -1, dtype=np.int32)
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            queue = [i]
            labels[i] = cluster_id
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                for nb in neighbors[cur]:
                    nb_i = int(nb)
                    if labels[nb_i] == -1:
                        labels[nb_i] = cluster_id
                        if is_core[nb_i]:
                            queue.append(nb_i)
            cluster_id += 1

        self.labels_ = vnp.array(labels, dtype=np.int32)
        self.core_sample_indices_ = vnp.array(np.asarray(core_list, dtype=np.int32), dtype=np.int32)
        return self

    def fit_predict(self, X: Any, y: Any = None) -> vnp.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
