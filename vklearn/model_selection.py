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


class KFold:
    """K-Fold cross-validation splitter yielding GPU index arrays."""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(self, X: Any, y: Any = None):
        n = len(X)
        idx = np.arange(n, dtype=np.int32)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(idx)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=np.int32)
        fold_sizes[: n % self.n_splits] += 1
        cur = 0
        for fs in fold_sizes:
            val_idx = idx[cur : cur + fs]
            tr_idx = np.concatenate([idx[:cur], idx[cur + fs :]])
            yield vnp.array(tr_idx, dtype=np.int32), vnp.array(val_idx, dtype=np.int32)
            cur += fs


def _clone_estimator(est: Any) -> Any:
    import inspect

    cls = type(est)
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if hasattr(est, name):
            val = getattr(est, name)
            if name == "steps" and isinstance(val, list):
                kwargs[name] = [(sn, _clone_estimator(se)) for sn, se in val]
            else:
                kwargs[name] = val
    return cls(**kwargs)


def cross_val_score(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    cv: Any = 5,
    scoring: Any = None,
) -> vnp.ndarray:
    """Evaluate a vklearn estimator by K-fold cross-validation, returning a GPU score array."""
    splitter = KFold(n_splits=int(cv), shuffle=True, random_state=42) if isinstance(cv, int) else cv
    scores: List[float] = []
    for tr_idx, val_idx in splitter.split(X, y):
        est = _clone_estimator(estimator)
        X_tr, X_val = _slice_rows(X, tr_idx), _slice_rows(X, val_idx)
        y_tr, y_val = _slice_rows(y, tr_idx), _slice_rows(y, val_idx)
        est.fit(X_tr, y_tr)
        if scoring is None:
            scores.append(float(est.score(X_val, y_val)))
        elif callable(scoring):
            scores.append(float(scoring(y_val, est.predict(X_val))))
        else:
            from . import metrics as _m

            metric_map = {
                "accuracy": _m.accuracy_score,
                "r2": _m.r2_score,
                "neg_mean_squared_error": lambda yt, yp: -_m.mean_squared_error(yt, yp),
                "neg_root_mean_squared_error": lambda yt, yp: -_m.root_mean_squared_error(yt, yp),
                "neg_mean_absolute_error": lambda yt, yp: -_m.mean_absolute_error(yt, yp),
            }
            if scoring not in metric_map:
                raise ValueError(f"Unsupported scoring={scoring!r}")
            scores.append(float(metric_map[scoring](y_val, est.predict(X_val))))
    return vnp.array(scores, dtype=np.float32)
