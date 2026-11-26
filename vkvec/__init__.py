"""vkvec: Pure-Vulkan GPU Vector Database, Cosine/L2 Search, & Text Vectorization."""

from .index import VectorIndex
from .similarity import chunk_text, cosine_similarity, dot_similarity, normalize
from .text import TfidfVectorizer

__all__ = [
    "TfidfVectorizer",
    "VectorIndex",
    "chunk_text",
    "cosine_similarity",
    "dot_similarity",
    "normalize",
]
