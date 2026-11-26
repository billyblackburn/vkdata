"""vkvec.hybrid: Reciprocal Rank Fusion (RRF) & Hybrid GPU Vector + BM25 Search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from .bm25 import BM25Index
from .index import VectorIndex
from .similarity import _to_gpu_2d


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Dict[str, Any]]],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Combine multiple ranked hit lists using Reciprocal Rank Fusion (RRF).

    For every document ``d`` appearing at 1-based rank ``r_i`` in list ``i``:

        ``RRF_score(d) = sum_i weights[i] / (k + r_i)``

    RRF is scale-invariant (it never breaks when BM25 scores are ``0..25`` and
    Cosine scores are ``-1..1``), making it the industry standard for Hybrid RAG.
    """
    n_lists = len(ranked_lists)
    if n_lists == 0:
        return []
    w = [1.0] * n_lists if weights is None else [float(x) for x in weights]
    if len(w) != n_lists:
        raise ValueError(f"weights length {len(w)} != number of ranked lists {n_lists}")

    fused_scores: Dict[Any, float] = {}
    meta_by_id: Dict[Any, Any] = {}
    first_seen: Dict[Any, int] = {}

    for list_idx, hits in enumerate(ranked_lists):
        weight = w[list_idx]
        if weight <= 0.0:
            continue
        for rank_0, hit in enumerate(hits):
            doc_id = hit["id"]
            if doc_id not in meta_by_id:
                meta_by_id[doc_id] = hit.get("metadata")
                first_seen[doc_id] = len(first_seen)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight / (float(k) + rank_0 + 1.0)

    ordered = sorted(fused_scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    if top_n is not None:
        ordered = ordered[: int(top_n)]

    return [
        {"id": doc_id, "score": float(score), "metadata": meta_by_id[doc_id]}
        for doc_id, score in ordered
    ]


class HybridIndex:
    """Production-style Hybrid Search combining Vulkan GPU ``VectorIndex`` + ``BM25Index``.

    Dense neural embeddings understand synonyms and intent; BM25 nails exact
    keywords, proper nouns, tickers, and error codes. ``HybridIndex`` runs both
    and fuses them via Reciprocal Rank Fusion (RRF) or weighted score blending.

    Examples
    --------
    >>> idx = HybridIndex(metric="cosine")
    >>> idx.add(embeddings, documents, metadata=[{"topic": "Tech"}, ...])
    >>> hits = idx.search(query_emb, "RTX4090 graphics card", k=5)
    """

    def __init__(
        self,
        metric: str = "cosine",
        *,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        stop_words: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
    ) -> None:
        self.vector_index = VectorIndex(metric=metric)
        self.bm25_index = BM25Index(k1=k1, b=b, stop_words=stop_words, min_token_len=min_token_len)
        self.rrf_k = int(rrf_k)

    def __len__(self) -> int:
        return len(self.vector_index)

    @property
    def dim(self) -> int:
        return self.vector_index.dim

    def add(
        self,
        vectors: Any,
        documents: Sequence[str],
        *,
        ids: Optional[Sequence[Union[int, str]]] = None,
        metadata: Optional[Sequence[Any]] = None,
    ) -> "HybridIndex":
        """Add paired vectors + raw text documents into both the GPU and BM25 indexes."""
        docs = list(documents)
        vg = _to_gpu_2d(vectors)
        if vg.shape[0] != len(docs):
            raise ValueError(f"vectors rows ({vg.shape[0]}) != number of documents ({len(docs)})")

        start_id = len(self.vector_index)
        shared_ids = list(ids) if ids is not None else list(range(start_id, start_id + len(docs)))
        shared_meta = list(metadata) if metadata is not None else [None] * len(docs)

        self.vector_index.add(vg, ids=shared_ids, metadata=shared_meta)
        self.bm25_index.add(docs, ids=shared_ids, metadata=shared_meta)
        return self

    def search_vector(
        self,
        query_vectors: Any,
        *,
        k: int = 5,
        where: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Search only the GPU dense vector index."""
        return self.vector_index.search(query_vectors, k=k, where=where)

    def search_bm25(
        self,
        query_texts: Union[str, Sequence[str]],
        *,
        k: int = 5,
        where: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Search only the BM25 full-text inverted index."""
        return self.bm25_index.search(query_texts, k=k, where=where)

    def search(
        self,
        query_vectors: Any,
        query_texts: Union[str, Sequence[str]],
        *,
        k: int = 5,
        where: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
        mode: str = "rrf",
        alpha: float = 0.5,
        candidate_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Run Hybrid Search fusing Vulkan GPU vector search + BM25 lexical search.

        Parameters
        ----------
        query_vectors : array-like
            1D ``(D,)`` or 2D ``(Q, D)`` query embeddings.
        query_texts : str or Sequence[str]
            Raw text query (or batch of ``Q`` queries) matching ``query_vectors``.
        k : int
            Number of fused results to return per query.
        where : dict or callable, optional
            Metadata filter applied consistently to both vector and BM25 search.
        mode : {'rrf', 'weighted'}
            ``'rrf'`` (default): Reciprocal Rank Fusion weighted by ``(alpha, 1 - alpha)``.
            ``'weighted'``: Min-max normalized linear blend ``alpha * vec + (1 - alpha) * bm25``.
        alpha : float
            Weight on dense vector search in ``[0, 1]`` (``1.0`` = pure vector, ``0.0`` = pure BM25).
        candidate_k : int, optional
            Number of candidates to retrieve from each index before fusion (defaults to ``max(k * 4, 20)``).
        rrf_k : int, optional
            Override the RRF rank smoothing constant (defaults to ``self.rrf_k``).
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        m = mode.lower()
        if m not in ("rrf", "weighted"):
            raise ValueError(f"Unsupported mode {mode!r}; expected 'rrf' or 'weighted'")

        q_texts = [query_texts] if isinstance(query_texts, str) else list(query_texts)
        if len(self) == 0:
            return [[] for _ in q_texts]

        qg = _to_gpu_2d(query_vectors)
        if qg.shape[0] != len(q_texts):
            raise ValueError(f"query_vectors rows ({qg.shape[0]}) != query_texts count ({len(q_texts)})")

        fetch_k = min(len(self), int(candidate_k) if candidate_k is not None else max(int(k) * 4, 20))
        vec_results = self.vector_index.search(qg, k=fetch_k, where=where)
        bm25_results = self.bm25_index.search(q_texts, k=fetch_k, where=where)

        rk = self.rrf_k if rrf_k is None else int(rrf_k)
        fused_all: List[List[Dict[str, Any]]] = []

        for v_hits, b_hits in zip(vec_results, bm25_results):
            if m == "rrf":
                fused_all.append(
                    reciprocal_rank_fusion(
                        [v_hits, b_hits],
                        k=rk,
                        weights=[alpha, 1.0 - alpha],
                        top_n=k,
                    )
                )
            else:
                fused_all.append(self._weighted_blend(v_hits, b_hits, alpha=alpha, top_n=k))

        return fused_all

    def _weighted_blend(
        self,
        v_hits: Sequence[Dict[str, Any]],
        b_hits: Sequence[Dict[str, Any]],
        *,
        alpha: float,
        top_n: int,
    ) -> List[Dict[str, Any]]:
        def _norm_scores(hits: Sequence[Dict[str, Any]], invert: bool = False) -> Dict[Any, float]:
            if not hits:
                return {}
            vals = [(-h["score"] if invert else h["score"]) for h in hits]
            lo, hi = min(vals), max(vals)
            span = hi - lo
            if span <= 1e-12:
                return {h["id"]: 1.0 for h in hits}
            return {h["id"]: (v - lo) / span for h, v in zip(hits, vals)}

        v_norm = _norm_scores(v_hits, invert=(self.vector_index.metric == "l2"))
        b_norm = _norm_scores(b_hits, invert=False)
        meta_by_id: Dict[Any, Any] = {}
        for h in list(v_hits) + list(b_hits):
            meta_by_id.setdefault(h["id"], h.get("metadata"))

        all_ids = list(meta_by_id.keys())
        scored = [
            {
                "id": doc_id,
                "score": float(alpha * v_norm.get(doc_id, 0.0) + (1.0 - alpha) * b_norm.get(doc_id, 0.0)),
                "metadata": meta_by_id[doc_id],
            }
            for doc_id in all_ids
        ]
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored[: int(top_n)]

    def save(self, path: Union[str, Path]) -> Path:
        """Persist both the GPU VectorIndex and BM25Index to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index.save(p)
        self.bm25_index.save(p.with_suffix(".bm25.json"))
        p.with_suffix(".hybrid.json").write_text(json.dumps({"rrf_k": self.rrf_k}), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HybridIndex":
        """Reload a saved ``HybridIndex`` directly into Vulkan GPU VRAM + BM25 index."""
        p = Path(path)
        cfg = json.loads(p.with_suffix(".hybrid.json").read_text(encoding="utf-8"))
        idx = cls(rrf_k=cfg["rrf_k"])
        idx.vector_index = VectorIndex.load(p)
        idx.bm25_index = BM25Index.load(p.with_suffix(".bm25.json"))
        return idx
