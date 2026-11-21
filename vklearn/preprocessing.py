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
