"""GPU-accelerated regression & classification metrics."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_1d


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    yt = to_gpu_1d(y_true, np.float32)
    yp = to_gpu_1d(y_pred, np.float32)
    diff = yt - yp
    return float(vnp.mean(diff * diff).item())


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    yt = to_gpu_1d(y_true, np.float32)
    yp = to_gpu_1d(y_pred, np.float32)
    return float(vnp.mean(vnp.abs(yt - yp)).item())


def r2_score(y_true: Any, y_pred: Any) -> float:
    yt = to_gpu_1d(y_true, np.float32)
    yp = to_gpu_1d(y_pred, np.float32)
    diff = yt - yp
    ss_res = float(vnp.sum(diff * diff).item())
    y_mean = vnp.mean(yt)
    centered = yt - y_mean
    ss_tot = float(vnp.sum(centered * centered).item())
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    yt = to_gpu_1d(y_true, np.int32)
    yp = to_gpu_1d(y_pred, np.int32)
    return float(vnp.mean((yt == yp).astype(np.float32)).item())


def confusion_matrix(y_true: Any, y_pred: Any, num_classes: Optional[int] = None) -> vnp.ndarray:
    """Compute (K, K) confusion matrix on GPU via zero-copy O(N) vknumpy.bincount."""
    yt = to_gpu_1d(y_true, np.int32)
    yp = to_gpu_1d(y_pred, np.int32)
    if num_classes is None:
        k = int(max(float(vnp.max(yt.astype(np.float32)).item()), float(vnp.max(yp.astype(np.float32)).item()))) + 1
    else:
        k = int(num_classes)
    bins = yt * k + yp
    return vnp.bincount(bins, minlength=k * k).reshape(k, k)
