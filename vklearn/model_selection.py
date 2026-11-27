"""GPU model selection helpers (train_test_split)."""

from __future__ import annotations

from typing import Any, List, Optional
import numpy as np
import vknumpy as vnp


def _slice_rows(arr: Any, idx: vnp.ndarray) -> Any:
    if hasattr(arr, "_take_rows"):
        return arr._take_rows(idx)
    if hasattr(arr, "_take"):
        return arr._take(idx)
    if isinstance(arr, vnp.ndarray):
        return vnp.take(arr, idx, axis=0)
    if isinstance(arr, list):
        return [arr[int(i)] for i in idx.to_numpy()]
    return np.asarray(arr)[idx.to_numpy()]


def train_test_split(
    *arrays: Any,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[Any]:
    """Split GPU vknumpy/vkpandas arrays into random train and test subsets."""
    if not arrays:
        raise ValueError("At least one array required")
    n = len(arrays[0])
    n_test = int(round(n * test_size)) if isinstance(test_size, float) else int(test_size)
    n_test = max(1, min(n - 1, n_test))
    n_train = n - n_test

    if shuffle:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n).astype(np.int32)
    else:
        perm = np.arange(n, dtype=np.int32)

    train_idx = vnp.array(perm[:n_train], dtype=np.int32)
    test_idx = vnp.array(perm[n_train:], dtype=np.int32)

    out: List[Any] = []
    for a in arrays:
        out.append(_slice_rows(a, train_idx))
        out.append(_slice_rows(a, test_idx))
    return out
