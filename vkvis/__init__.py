"""vkvis: mlx-vis-style Vulkan GPU dimensionality reduction & neighbor embedding."""

from .cne import CNE
from .knn import NNDescent, brute_force_knn, knn_search
from .pacmap import LocalMAP, PaCMAP
from .render import animate_embedding, compare_embeddings, render_embedding
from .trimap import TriMap
from .tsne import DREAMS, TSNE
from .umap import UMAP

__all__ = [
    "CNE",
    "DREAMS",
    "LocalMAP",
    "NNDescent",
    "PaCMAP",
    "TSNE",
    "TriMap",
    "UMAP",
    "animate_embedding",
    "brute_force_knn",
    "compare_embeddings",
    "knn_search",
    "render_embedding",
]
