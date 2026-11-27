"""vkvec.ivf: GPU Inverted File (IVF-Flat) Approximate Nearest Neighbor index (like cuVS / FAISS)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

import vknumpy as vnp
from .similarity import normalize


def _to_2d_np(x: Any) -> np.ndarray:
    if isinstance(x, vnp.ndarray):
        arr = x.to_numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D vectors, got shape {arr.shape}")
    return arr


def _match_where(meta: Dict[str, Any], where: Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]) -> bool:
    if callable(where):
        return bool(where(meta))
    for k, v in where.items():
        if meta.get(k) != v:
            return False
    return True


class IVFIndex:
    """Vulkan GPU Inverted File (IVF-Flat) ANN index for fast large-scale vector search.

    Partitions the vector space into ``nlist`` Voronoi cells via GPU K-Means, then probes
    only the ``nprobe`` closest cells at query time (skipping 85–95%+ of the database).
    """

    def __init__(
        self,
        nlist: int = 64,
        nprobe: int = 8,
        metric: str = "cosine",
        seed: int = 42,
    ) -> None:
        if metric not in ("cosine", "l2", "dot"):
            raise ValueError(f"metric must be 'cosine', 'l2', or 'dot', got {metric!r}")
        if nlist < 1:
            raise ValueError(f"nlist must be >= 1, got {nlist}")
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.metric = metric
        self.seed = int(seed)

        self.dim: Optional[int] = None
        self.centroids_: Optional[vnp.ndarray] = None  # (nlist_eff, D) on GPU
        self.is_trained: bool = False

        # Inverted lists: list_id -> list of global vector indices
        self._lists: List[List[int]] = []
        self._list_vecs_gpu: List[Optional[vnp.ndarray]] = []
        self._vecs_np: Optional[np.ndarray] = None
        self._ids: List[Any] = []
        self._metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._ids)

    def _prep_gpu(self, arr_np: np.ndarray) -> vnp.ndarray:
        g = vnp.array(arr_np, dtype=np.float32)
        if self.metric == "cosine":
            g = normalize(g)
        return g

    def _score_against(self, q_gpu: vnp.ndarray, mat_gpu: vnp.ndarray) -> vnp.ndarray:
        """Higher-is-better score matrix (Q, M) on Vulkan GPU."""
        if self.metric in ("cosine", "dot"):
            return q_gpu @ mat_gpu.T
        # Negative squared L2 distance (higher = closer)
        q2 = vnp.sum(q_gpu * q_gpu, axis=1).reshape(-1, 1)
        m2 = vnp.sum(mat_gpu * mat_gpu, axis=1).reshape(1, -1)
        d2 = vnp.maximum(q2 + m2 - 2.0 * (q_gpu @ mat_gpu.T), 0.0)
        return -d2

    def train(self, vectors: Any, max_iter: int = 25) -> "IVFIndex":
        """Train coarse Voronoi centroids on Vulkan GPU using vklearn.KMeans."""
        from vklearn.cluster import KMeans

        arr_np = _to_2d_np(vectors)
        n, d = arr_np.shape
        self.dim = d
        k = min(self.nlist, n)
        g = self._prep_gpu(arr_np)
        km = KMeans(n_clusters=k, max_iter=max_iter, random_state=self.seed).fit(g)
        c = km.cluster_centers_
        assert c is not None
        if self.metric == "cosine":
            c = normalize(c)
        self.centroids_ = c
        self._lists = [[] for _ in range(k)]
        self._list_vecs_gpu = [None for _ in range(k)]
        self.is_trained = True
        return self

    def add(
        self,
        vectors: Any,
        ids: Optional[Sequence[Any]] = None,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "IVFIndex":
        """Assign vectors to their nearest coarse GPU centroid and append to inverted lists."""
        arr_np = _to_2d_np(vectors)
        n, d = arr_np.shape
        if not self.is_trained:
            self.train(arr_np)
        assert self.centroids_ is not None and self.dim is not None
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")

        g = self._prep_gpu(arr_np)
        stored_np = g.to_numpy() if self.metric == "cosine" else arr_np

        base_idx = len(self._ids)
        if ids is None:
            new_ids = list(range(base_idx, base_idx + n))
        else:
            new_ids = list(ids)
            if len(new_ids) != n:
                raise ValueError(f"len(ids)={len(new_ids)} != n_vectors={n}")
        if metadata is None:
            new_meta = [{} for _ in range(n)]
        else:
            new_meta = [dict(m) for m in metadata]
            if len(new_meta) != n:
                raise ValueError(f"len(metadata)={len(new_meta)} != n_vectors={n}")

        self._ids.extend(new_ids)
        self._metadata.extend(new_meta)
        self._vecs_np = stored_np if self._vecs_np is None else np.vstack([self._vecs_np, stored_np])

        # Chunked GPU assignment to nearest coarse centroid
        assign_chunks: List[np.ndarray] = []
        for s in range(0, n, 4096):
            scores = self._score_against(g[s : s + 4096], self.centroids_)
            assign_chunks.append(vnp.argmax(scores, axis=1).to_numpy().astype(np.int32))
        assigns = np.concatenate(assign_chunks)

        touched = set()
        for local_i, cell in enumerate(assigns):
            c_i = int(cell)
            self._lists[c_i].append(base_idx + local_i)
            touched.add(c_i)

        # Refresh GPU buffers for touched inverted lists
        for c_i in touched:
            idxs = self._lists[c_i]
            self._list_vecs_gpu[c_i] = vnp.array(self._vecs_np[idxs], dtype=np.float32)

        return self

    def search(
        self,
        query: Any,
        k: int = 10,
        *,
        nprobe: Optional[int] = None,
        where: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]] = None,
    ) -> Any:
        """Search top-k nearest neighbors by probing only the ``nprobe`` closest GPU cells."""
        if not self.is_trained or self.centroids_ is None or len(self._ids) == 0:
            return []
        raw_q = query.to_numpy() if isinstance(query, vnp.ndarray) else np.asarray(query, dtype=np.float32)
        single = raw_q.ndim == 1
        q_np = _to_2d_np(raw_q)
        q_gpu = self._prep_gpu(q_np)

        n_cells = self.centroids_.shape[0]
        probe_k = max(1, min(int(nprobe if nprobe is not None else self.nprobe), n_cells))

        # 1. Coarse search on Vulkan GPU: pick top-nprobe centroids per query
        c_scores = self._score_against(q_gpu, self.centroids_).to_numpy()
        all_results: List[List[Dict[str, Any]]] = []

        for qi in range(q_np.shape[0]):
            top_cells = np.argsort(-c_scores[qi])[:probe_k]
            cand_ids: List[int] = []
            cand_scores: List[np.ndarray] = []
            qi_gpu = q_gpu[qi : qi + 1]

            for cell in top_cells:
                c_i = int(cell)
                cell_ids = self._lists[c_i]
                cell_mat = self._list_vecs_gpu[c_i]
                if not cell_ids or cell_mat is None:
                    continue
                sc = self._score_against(qi_gpu, cell_mat).to_numpy().reshape(-1)
                cand_ids.extend(cell_ids)
                cand_scores.append(sc)

            if not cand_ids:
                all_results.append([])
                continue

            scores_flat = np.concatenate(cand_scores)
            order = np.argsort(-scores_flat)
            hits: List[Dict[str, Any]] = []
            for pos in order:
                gidx = cand_ids[int(pos)]
                meta = self._metadata[gidx]
                if where is not None and not _match_where(meta, where):
                    continue
                score_val = float(scores_flat[int(pos)])
                if self.metric == "l2":
                    score_val = float(np.sqrt(max(0.0, -score_val)))
                hits.append({"id": self._ids[gidx], "score": score_val, "metadata": meta})
                if len(hits) >= k:
                    break
            all_results.append(hits)

        return all_results[0] if single else all_results

    def save(self, path: Union[str, Path]) -> Path:
        """Persist IVF centroids, vectors, inverted lists, and metadata to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        npz_path = p.with_suffix(".npz")
        meta_path = p.with_suffix(".ivf.json")
        np.savez(
            npz_path,
            centroids=self.centroids_.to_numpy() if self.centroids_ is not None else np.empty((0, 0), dtype=np.float32),
            vectors=self._vecs_np if self._vecs_np is not None else np.empty((0, 0), dtype=np.float32),
        )
        payload = {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "metric": self.metric,
            "seed": self.seed,
            "dim": self.dim,
            "lists": self._lists,
            "ids": self._ids,
            "metadata": self._metadata,
        }
        meta_path.write_text(json.dumps(payload), encoding="utf-8")
        return npz_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IVFIndex":
        """Reload a saved IVFIndex directly onto the Vulkan GPU."""
        p = Path(path)
        npz_path = p.with_suffix(".npz")
        meta_path = p.with_suffix(".ivf.json")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(npz_path)

        idx = cls(
            nlist=payload["nlist"],
            nprobe=payload["nprobe"],
            metric=payload["metric"],
            seed=payload["seed"],
        )
        idx.dim = payload["dim"]
        idx._lists = payload["lists"]
        idx._ids = payload["ids"]
        idx._metadata = payload["metadata"]
        cents = data["centroids"]
        vecs = data["vectors"]
        if cents.size > 0:
            idx.centroids_ = vnp.array(cents.astype(np.float32))
            idx.is_trained = True
        if vecs.size > 0:
            idx._vecs_np = vecs.astype(np.float32)
            idx._list_vecs_gpu = [
                vnp.array(idx._vecs_np[cell_ids], dtype=np.float32) if cell_ids else None
                for cell_ids in idx._lists
            ]
        else:
            idx._list_vecs_gpu = [None for _ in idx._lists]
        return idx


__all__ = ["IVFIndex"]
