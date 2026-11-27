"""GPU preprocessing transformers (StandardScaler, MinMaxScaler)."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_2d


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance on Vulkan GPU."""

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.mean_: Optional[vnp.ndarray] = None
        self.scale_: Optional[vnp.ndarray] = None
        self.var_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "StandardScaler":
        Xg = to_gpu_2d(X)
        self.mean_ = vnp.mean(Xg, axis=0) if self.with_mean else vnp.zeros((Xg.shape[1],), dtype=np.float32)
        if self.with_std:
            self.var_ = vnp.var(Xg, axis=0)
            std = vnp.sqrt(self.var_)
            self.scale_ = vnp.where(std == 0.0, 1.0, std)
        else:
            self.var_ = None
            self.scale_ = vnp.ones((Xg.shape[1],), dtype=np.float32)
        return self

    def transform(self, X: Any) -> vnp.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform()")
        Xg = to_gpu_2d(X)
        out = Xg - self.mean_ if self.with_mean else Xg
        return out / self.scale_ if self.with_std else out

    def fit_transform(self, X: Any, y: Any = None) -> vnp.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> vnp.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling inverse_transform()")
        Xg = to_gpu_2d(X)
        out = Xg * self.scale_ if self.with_std else Xg
        return out + self.mean_ if self.with_mean else out


class MinMaxScaler:
    """Scale features to a target [min, max] range on Vulkan GPU."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = (float(feature_range[0]), float(feature_range[1]))
        self.data_min_: Optional[vnp.ndarray] = None
        self.data_max_: Optional[vnp.ndarray] = None
        self.data_range_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "MinMaxScaler":
        Xg = to_gpu_2d(X)
        self.data_min_ = vnp.min(Xg, axis=0)
        self.data_max_ = vnp.max(Xg, axis=0)
        dr = self.data_max_ - self.data_min_
        self.data_range_ = vnp.where(dr == 0.0, 1.0, dr)
        return self

    def transform(self, X: Any) -> vnp.ndarray:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before calling transform()")
        Xg = to_gpu_2d(X)
        lo, hi = self.feature_range
        std = (Xg - self.data_min_) / self.data_range_
        return std * (hi - lo) + lo

    def fit_transform(self, X: Any, y: Any = None) -> vnp.ndarray:
        return self.fit(X, y).transform(X)


class SimpleImputer:
    """Leak-free NaN imputer (median, mean, or constant) compatible with vklearn.Pipeline."""

    def __init__(self, strategy: str = "median", fill_value: float = 0.0) -> None:
        s = strategy.lower()
        if s not in ("median", "mean", "constant"):
            raise ValueError(f"Unsupported strategy {strategy!r}; expected 'median', 'mean', or 'constant'")
        self.strategy = s
        self.fill_value = float(fill_value)
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "SimpleImputer":
        arr = to_gpu_2d(X).to_numpy()
        if self.strategy == "median":
            stats = np.nanmedian(arr, axis=0)
        elif self.strategy == "mean":
            stats = np.nanmean(arr, axis=0)
        else:
            stats = np.full((arr.shape[1],), self.fill_value, dtype=np.float32)
        self.statistics_ = np.where(np.isnan(stats), self.fill_value, stats).astype(np.float32)
        return self

    def transform(self, X: Any) -> vnp.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before calling transform()")
        was_1d = getattr(X, "ndim", 2) == 1
        arr = to_gpu_2d(X).to_numpy().copy()
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rows, cols = np.nonzero(nan_mask)
            arr[rows, cols] = self.statistics_[cols]
        return vnp.array(arr.ravel() if was_1d else arr, dtype=np.float32)

    def fit_transform(self, X: Any, y: Any = None) -> vnp.ndarray:
        return self.fit(X, y).transform(X)


class OneHotEncoder:
    """Encode 1D categorical strings/integers into float32 one-hot columns on Vulkan GPU."""

    def __init__(self, *, drop_last: bool = False) -> None:
        self.drop_last = bool(drop_last)
        self.categories_: Optional[list] = None

    def fit(self, x: Any, y: Any = None) -> "OneHotEncoder":
        arr = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
        self.categories_ = sorted({str(v) for v in arr.ravel()})
        return self

    def transform(self, x: Any) -> vnp.ndarray:
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before calling transform()")
        arr = (x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)).astype(str).ravel()
        cats = self.categories_[:-1] if (self.drop_last and len(self.categories_) > 1) else self.categories_
        cols = [(arr == c).astype(np.float32) for c in cats]
        mat = np.column_stack(cols) if cols else np.zeros((len(arr), 0), dtype=np.float32)
        return vnp.array(np.ascontiguousarray(mat), dtype=np.float32)

    def fit_transform(self, x: Any, y: Any = None) -> vnp.ndarray:
        return self.fit(x, y).transform(x)
