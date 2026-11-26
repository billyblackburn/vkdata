"""vkvec.bm25: Okapi BM25 Full-Text Inverted Index with Metadata Filtering."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .text import DEFAULT_STOP_WORDS


class BM25Index:
    """In-memory Okapi BM25 full-text inverted index for exact keyword & lexical search.

    Designed to pair with :class:`vkvec.VectorIndex` inside :class:`vkvec.HybridIndex`
    so rare proper nouns, error codes, product IDs, and tickers match with 100% precision.

    Parameters
    ----------
    k1 : float
        Term-frequency saturation parameter (default ``1.5``).
    b : float
        Document-length normalization strength in ``[0, 1]`` (default ``0.75``).
    stop_words : Optional[Iterable[str]]
        Words to ignore (defaults to ``DEFAULT_STOP_WORDS``; pass ``()`` to keep all tokens).
    min_token_len : int
        Minimum alphanumeric token length to index (default ``2`` so tickers/codes like ``Q3`` match).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        *,
        stop_words: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.min_token_len = int(min_token_len)
        self.stop_words: Set[str] = DEFAULT_STOP_WORDS if stop_words is None else set(stop_words)
        self._tok_re = re.compile(rf"[a-z0-9]{{{self.min_token_len},}}")

        self._ids: List[Union[int, str]] = []
        self._metadata: List[Any] = []
        self._documents: List[str] = []
        self._doc_lens: List[int] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._total_tokens: int = 0

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    @property
    def avgdl(self) -> float:
        return (self._total_tokens / len(self._ids)) if self._ids else 0.0

    def tokenize(self, text: str) -> List[str]:
        return [w for w in self._tok_re.findall(text.lower()) if w not in self.stop_words]

    def add(
        self,
        documents: Sequence[str],
        *,
        ids: Optional[Sequence[Union[int, str]]] = None,
        metadata: Optional[Sequence[Any]] = None,
    ) -> "BM25Index":
        """Index text documents (with optional IDs & metadata) into the inverted index."""
        docs = list(documents)
        n = len(docs)
        start_idx = len(self._ids)
        new_ids = list(ids) if ids is not None else list(range(start_idx, start_idx + n))
        new_meta = list(metadata) if metadata is not None else [None] * n
        if len(new_ids) != n or len(new_meta) != n:
            raise ValueError("ids and metadata must match the number of documents")

        for offset, doc in enumerate(docs):
            doc_idx = start_idx + offset
            tokens = self.tokenize(doc)
            self._doc_lens.append(len(tokens))
            self._total_tokens += len(tokens)
            counts = Counter(tokens)
            for term, tf in counts.items():
                self._postings[term].append((doc_idx, int(tf)))

        self._ids.extend(new_ids)
        self._metadata.extend(new_meta)
        self._documents.extend(docs)
        return self

    def _allowed_mask(
        self,
        where: Optional[Union[Dict[str, Any], Callable[[Any], bool]]],
    ) -> Optional[Set[int]]:
        if where is None:
            return None
        if callable(where):
            return {i for i, m in enumerate(self._metadata) if bool(where(m))}
        return {
            i
            for i, m in enumerate(self._metadata)
            if isinstance(m, dict) and all(m.get(wk) == wv for wk, wv in where.items())
        }

    def search(
        self,
        queries: Union[str, Sequence[str]],
        *,
        k: int = 5,
        where: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Search the BM25 inverted index and return top-``k`` hits per query.

        Each hit is a dict: ``{"id": ..., "score": float, "metadata": ...}``.
        """
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        n_docs = len(self._ids)
        if n_docs == 0:
            return [[] for _ in query_list]

        allowed = self._allowed_mask(where)
        if allowed is not None and len(allowed) == 0:
            return [[] for _ in query_list]

        avgdl = self.avgdl or 1.0
        k1 = self.k1
        b = self.b
        results: List[List[Dict[str, Any]]] = []

        for q in query_list:
            q_tokens = self.tokenize(q)
            if not q_tokens:
                results.append([])
                continue

            scores: Dict[int, float] = defaultdict(float)
            for term in set(q_tokens):
                posting = self._postings.get(term)
                if not posting:
                    continue
                df = len(posting)
                # Lucene-style non-negative BM25 IDF
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                for doc_idx, tf in posting:
                    if allowed is not None and doc_idx not in allowed:
                        continue
                    dl = self._doc_lens[doc_idx]
                    denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                    scores[doc_idx] += idf * (tf * (k1 + 1.0) / denom)

            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: int(k)]
            results.append(
                [
                    {"id": self._ids[doc_idx], "score": float(score), "metadata": self._metadata[doc_idx]}
                    for doc_idx, score in ranked
                ]
            )
        return results

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the BM25 index (documents, IDs, metadata, and hyperparameters) to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "min_token_len": self.min_token_len,
            "stop_words": sorted(self.stop_words),
            "ids": self._ids,
            "metadata": self._metadata,
            "documents": self._documents,
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Index":
        """Reload a saved ``BM25Index`` from disk."""
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        idx = cls(
            k1=payload["k1"],
            b=payload["b"],
            stop_words=payload["stop_words"],
            min_token_len=payload["min_token_len"],
        )
        if payload["documents"]:
            idx.add(payload["documents"], ids=payload["ids"], metadata=payload["metadata"])
        return idx
