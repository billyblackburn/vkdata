"""vkvec.text: GPU-accelerated TF-IDF & Bag-of-Words text-to-vector encoders."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import vknumpy as vnp

from .similarity import normalize

DEFAULT_STOP_WORDS: Set[str] = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was", "were",
    "has", "have", "had", "its", "his", "her", "their", "but", "not", "will",
    "would", "about", "into", "over", "after", "more", "than", "new", "said",
    "says", "two", "one", "out", "who", "they", "been", "all", "can", "year",
}


class TfidfVectorizer:
    """Turn text documents into L2-normalized GPU TF-IDF (or Bag-of-Words) vectors.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size (keeps top most-frequent terms in training docs).
    use_idf : bool
        If True, applies TF-IDF weighting; if False, plain Bag-of-Words counts.
    sublinear_tf : bool
        If True, uses ``log(1 + count)`` term frequency scaling.
    l2_normalize : bool
        If True, L2-normalizes every output vector on Vulkan GPU so dot product == Cosine Similarity.
    stop_words : Optional[Iterable[str]]
        Words to ignore (defaults to ``DEFAULT_STOP_WORDS``; pass ``()`` to keep all words).
    min_token_len : int
        Minimum alphabetic token length to keep (default 3).
    """

    def __init__(
        self,
        max_features: int = 2000,
        *,
        use_idf: bool = True,
        sublinear_tf: bool = True,
        l2_normalize: bool = True,
        stop_words: Optional[Iterable[str]] = None,
        min_token_len: int = 3,
    ) -> None:
        self.max_features = int(max_features)
        self.use_idf = bool(use_idf)
        self.sublinear_tf = bool(sublinear_tf)
        self.l2_normalize = bool(l2_normalize)
        self.stop_words: Set[str] = DEFAULT_STOP_WORDS if stop_words is None else set(stop_words)
        self._tok_re = re.compile(rf"[a-z]{{{int(min_token_len)},}}")

        self.vocabulary_: List[str] = []
        self.word2id_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None

    def tokenize(self, text: str) -> List[str]:
        return [w for w in self._tok_re.findall(text.lower()) if w not in self.stop_words]

    def fit(self, documents: Sequence[str]) -> "TfidfVectorizer":
        doc_freq: Counter = Counter()
        n_docs = len(documents)
        for doc in documents:
            doc_freq.update(set(self.tokenize(doc)))

        self.vocabulary_ = [w for w, _ in doc_freq.most_common(self.max_features)]
        self.word2id_ = {w: i for i, w in enumerate(self.vocabulary_)}
        self.idf_ = np.array(
            [np.log((1.0 + n_docs) / (1.0 + doc_freq[w])) + 1.0 for w in self.vocabulary_],
            dtype=np.float32,
        )
        return self

    def transform(self, documents: Sequence[str]) -> vnp.ndarray:
        if not self.vocabulary_ or self.idf_ is None:
            raise RuntimeError("TfidfVectorizer must be fitted before transform()")
        dim = len(self.vocabulary_)
        mat = np.zeros((len(documents), dim), dtype=np.float32)
        w2i = self.word2id_
        for r, doc in enumerate(documents):
            for w in self.tokenize(doc):
                idx = w2i.get(w)
                if idx is not None:
                    mat[r, idx] += 1.0

        if self.sublinear_tf:
            mat = np.log1p(mat)
        if self.use_idf:
            mat = mat * self.idf_[None, :]

        gpu_mat = vnp.array(mat, dtype=np.float32)
        return normalize(gpu_mat) if self.l2_normalize else gpu_mat

    def fit_transform(self, documents: Sequence[str]) -> vnp.ndarray:
        return self.fit(documents).transform(documents)
