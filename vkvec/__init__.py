"""vkvec: Pure-Vulkan GPU Vector Database, BM25 Full-Text & Hybrid Search, & Text Vectorization."""

from .bm25 import BM25Index
from .datasets import BBC_NEWS_CLASSES, load_bbc_news
from .hybrid import HybridIndex, reciprocal_rank_fusion
from .index import VectorIndex
from .ivf import IVFIndex
from .similarity import chunk_text, cosine_similarity, dot_similarity, normalize
from .text import TfidfVectorizer

__all__ = [
    "BBC_NEWS_CLASSES",
    "BM25Index",
    "HybridIndex",
    "IVFIndex",
    "TfidfVectorizer",
    "VectorIndex",
    "chunk_text",
    "cosine_similarity",
    "dot_similarity",
    "load_bbc_news",
    "normalize",
    "reciprocal_rank_fusion",
]
