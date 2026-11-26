"""vkvec.index: Pure-Vulkan GPU VectorIndex (FAISS / Chroma-style VRAM vector database)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import vknumpy as vnp

from .similarity import _to_gpu_2d, normalize


class VectorIndex:
    """In-VRAM Vulkan GPU Vector Database supporting Cosine & L2 similarity search.

    Examples
    --------
    >>> idx = VectorIndex(metric="cosine")
    >>> idx.add(vectors, metadata=[{"title": "Doc 1"}, {"title": "Doc 2"}])
    >>> hits = idx.search(query_vec, k=3)
    """

    def __init__(self, metric: str = "cosine") -> None:
        m = metric.lower()
        if m not in ("cosine", "l2", "dot"):
            raise ValueError(f"Unsupported metric {metric!r}; expected 'cosine', 'l2', or 'dot'")
        self.metric = m
        self._vectors: Optional[vnp.ndarray] = None
        self._norms_sq: Optional[vnp.ndarray] = None
        self._ids: List[Union[int, str]] = []
        self._metadata: List[Any] = []

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[1])

    def add(
        self,
        vectors: Any,
        *,
        ids: Optional[Sequence[Union[int, str]]] = None,
        metadata: Optional[Sequence[Any]] = None,
    ) -> "VectorIndex":
        """Add vectors (and optional IDs & metadata) into the GPU index."""
        vg = _to_gpu_2d(vectors)
        n, d = vg.shape
        if self._vectors is not None and d != self.dim:
            raise ValueError(f"Dimension mismatch: index has dim={self.dim}, got {d}")

        if self.metric == "cosine":
            vg = normalize(vg)

        start_id = len(self._ids)
        new_ids = list(ids) if ids is not None else list(range(start_id, start_id + n))
        new_meta = list(metadata) if metadata is not None else [None] * n
        if len(new_ids) != n or len(new_meta) != n:
            raise ValueError("ids and metadata must match the number of vectors")

        if self._vectors is None:
            self._vectors = vg.contiguous()
        else:
            self._vectors = vnp.concatenate([self._vectors, vg], axis=0).contiguous()

        if self.metric == "l2":
            self._norms_sq = vnp.sum(self._vectors * self._vectors, axis=1).reshape(1, -1)

        self._ids.extend(new_ids)
        self._metadata.extend(new_meta)
        return self

    def search(
        self,
        queries: Any,
        *,
        k: int = 5,
        where: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Search the GPU vector index and return top-``k`` hits per query.

        Each hit is a dict: ``{"id": ..., "score": float, "metadata": ...}``.
        For ``metric='cosine'`` / ``'dot'``, higher ``score`` is better.
        For ``metric='l2'``, ``score`` is Euclidean distance (lower is better).
        """
        if self._vectors is None or len(self._ids) == 0:
            return []

        qg = _to_gpu_2d(queries)
        if qg.shape[1] != self.dim:
            raise ValueError(f"Query dim {qg.shape[1]} != index dim {self.dim}")

        if self.metric == "cosine":
            qg = normalize(qg)
            scores_gpu = qg @ self._vectors.T.contiguous()  # higher is better
        elif self.metric == "dot":
            scores_gpu = qg @ self._vectors.T.contiguous()
        else:
            # Squared L2: ||q||^2 + ||x||^2 - 2 q @ x.T; negate so top_k picks smallest distances
            q_sq = vnp.sum(qg * qg, axis=1, keepdims=True)
            assert self._norms_sq is not None
            dist_sq = vnp.maximum(q_sq + self._norms_sq - (qg @ self._vectors.T.contiguous()) * 2.0, 0.0)
            scores_gpu = -vnp.sqrt(dist_sq)

        n_total = len(self._ids)
        # Apply optional metadata filter mask before top-k
        if where is not None:
            if callable(where):
                mask_np = np.array([bool(where(m)) for m in self._metadata], dtype=np.float32)
            else:
                mask_np = np.array(
                    [isinstance(m, dict) and all(m.get(wk) == wv for wk, wv in where.items()) for m in self._metadata],
                    dtype=np.float32,
                )
            n_valid = int(mask_np.sum())
            if n_valid == 0:
                return [[] for _ in range(qg.shape[0])]
            penalty = vnp.array(np.where(mask_np > 0.5, 0.0, -1e9).astype(np.float32)).reshape(1, n_total)
            scores_gpu = scores_gpu + penalty
            k_eff = min(int(k), n_valid)
        else:
            k_eff = min(int(k), n_total)

        top_idx = vnp.top_k(scores_gpu, k=k_eff, axis=1).to_numpy()
        scores_np = scores_gpu.to_numpy()

        results: List[List[Dict[str, Any]]] = []
        for qi in range(qg.shape[0]):
            row_hits: List[Dict[str, Any]] = []
            for idx in top_idx[qi]:
                i = int(idx)
                raw_s = float(scores_np[qi, i])
                score = (-raw_s) if self.metric == "l2" else raw_s
                row_hits.append({"id": self._ids[i], "score": score, "metadata": self._metadata[i]})
            # Sort deterministically (higher better for cosine/dot, lower better for l2)
            row_hits.sort(key=lambda h: h["score"], reverse=(self.metric != "l2"))
            results.append(row_hits)
        return results

    def save(self, path: Union[str, Path]) -> Path:
        """Persist vectors (.npz) and metadata (.json) to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        vec_np = np.zeros((0, 0), dtype=np.float32) if self._vectors is None else self._vectors.to_numpy()
        np.savez_compressed(p, vectors=vec_np)
        meta_path = p.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps({"metric": self.metric, "ids": self._ids, "metadata": self._metadata}),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorIndex":
        """Reload a saved ``VectorIndex`` directly into Vulkan GPU VRAM."""
        p = Path(path)
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        idx = cls(metric=meta["metric"])
        vec_np = np.load(p)["vectors"]
        if vec_np.size > 0:
            idx.add(vec_np, ids=meta["ids"], metadata=meta["metadata"])
        return idx
