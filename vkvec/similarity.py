"""vkvec.similarity: Vulkan GPU vector normalization, similarity matrices, & RAG text chunking."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import vknumpy as vnp


def _to_gpu_2d(x: Any) -> vnp.ndarray:
    arr = x if isinstance(x, vnp.ndarray) else vnp.array(np.asarray(x, dtype=np.float32), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.contiguous()


def normalize(x: Any, eps: float = 1e-8) -> vnp.ndarray:
    """L2-normalize rows of ``x`` on Vulkan GPU so every vector has unit length (||v|| = 1)."""
    xg = _to_gpu_2d(x)
    norms = vnp.sqrt(vnp.sum(xg * xg, axis=1, keepdims=True) + float(eps))
    return xg / norms


def dot_similarity(a: Any, b: Any) -> vnp.ndarray:
    """Pairwise dot product ``a @ b.T`` on Vulkan GPU (ggml-vulkan GEMM)."""
    ag = _to_gpu_2d(a)
    bg = _to_gpu_2d(b)
    return ag @ bg.T.contiguous()


def cosine_similarity(a: Any, b: Any, eps: float = 1e-8) -> vnp.ndarray:
    """Pairwise Cosine Similarity matrix ``normalize(a) @ normalize(b).T`` on Vulkan GPU."""
    return normalize(a, eps=eps) @ normalize(b, eps=eps).T.contiguous()


def chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """Split a long document into overlapping word chunks (standard RAG preprocessing)."""
    words = text.split()
    if not words:
        return []
    step = max(1, int(chunk_size) - int(overlap))
    chunks: List[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + int(chunk_size)]
        if piece:
            chunks.append(" ".join(piece))
        if start + int(chunk_size) >= len(words):
            break
    return chunks
