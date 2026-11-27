"""Built-in GPU dataset loaders for vklearn."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

import vknumpy as vnp

FASHION_MNIST_CLASSES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

_FASHION_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
)
_FASHION_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _load_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return (
            np.frombuffer(f.read(), dtype=np.uint8)
            .reshape(n, rows, cols)
            .astype(np.float32)
            / 255.0
        )


def _load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


def _stratified_subset(
    X: np.ndarray, y: np.ndarray, per_class: int | None, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if per_class is None:
        return X, y
    rng = np.random.default_rng(seed)
    sel = np.concatenate(
        [
            rng.choice(np.where(y == c)[0], size=per_class, replace=False)
            for c in range(10)
        ]
    )
    rng.shuffle(sel)
    return X[sel], y[sel]


def load_fashion_mnist(
    data_dir: str | Path | None = None,
    per_class_train: int | None = 1200,
    per_class_test: int | None = 200,
    seed: int = 42,
):
    """Download/cache Zalando Fashion-MNIST and return GPU arrays on Vulkan.

    Returns ``(X_train, y_train, X_test, y_test, class_names)`` where images have
    shape ``(N, 28, 28)`` in ``[0, 1]`` and labels have shape ``(N,)`` as
    ``vknumpy.ndarray`` tensors.
    """
    if data_dir is None:
        repo_cache = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "tldr_end2end"
            / "data"
            / "fashion_mnist"
        )
        cache_dir = repo_cache if repo_cache.exists() else Path("data/fashion_mnist")
    else:
        cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for fn in _FASHION_FILES:
        fp = cache_dir / fn
        if not fp.exists():
            urllib.request.urlretrieve(_FASHION_BASE_URL + fn, fp)

    X_tr, y_tr = _stratified_subset(
        _load_idx_images(cache_dir / "train-images-idx3-ubyte.gz"),
        _load_idx_labels(cache_dir / "train-labels-idx1-ubyte.gz"),
        per_class=per_class_train,
        seed=seed,
    )
    X_te, y_te = _stratified_subset(
        _load_idx_images(cache_dir / "t10k-images-idx3-ubyte.gz"),
        _load_idx_labels(cache_dir / "t10k-labels-idx1-ubyte.gz"),
        per_class=per_class_test,
        seed=seed,
    )
    return (
        vnp.array(X_tr),
        vnp.array(y_tr),
        vnp.array(X_te),
        vnp.array(y_te),
        list(FASHION_MNIST_CLASSES),
    )


__all__ = ["FASHION_MNIST_CLASSES", "load_fashion_mnist"]
