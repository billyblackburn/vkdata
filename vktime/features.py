"""vktime.features: GPU rolling windows, lag matrices, sliding windows, & leak-free temporal splits."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import vknumpy as vnp


def _to_np_f32(x: Any) -> np.ndarray:
    if isinstance(x, vnp.ndarray):
        return x.to_numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _window_view_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Build causal (leading-edge-padded) sliding windows of length ``window`` for a 1D array."""
    w = int(window)
    if w < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if w == 1:
        return arr[:, None]
    pad = np.repeat(arr[:1], w - 1)
    padded = np.concatenate([pad, arr])
    return np.lib.stride_tricks.sliding_window_view(padded, w)


def rolling_mean(x: Any, window: int) -> vnp.ndarray:
    """Causal GPU rolling mean (same length as input, edge-padded at start)."""
    arr = _to_np_f32(x)
    if arr.ndim == 1:
        win = _window_view_1d(arr, window)
        return vnp.mean(vnp.array(np.ascontiguousarray(win), dtype=np.float32), axis=1)
    if arr.ndim == 2:
        cols = [rolling_mean(arr[:, c], window).to_numpy() for c in range(arr.shape[1])]
        return vnp.array(np.stack(cols, axis=1), dtype=np.float32)
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def rolling_std(x: Any, window: int, *, eps: float = 1e-8) -> vnp.ndarray:
    """Causal GPU rolling standard deviation (same length as input)."""
    arr = _to_np_f32(x)
    if arr.ndim == 1:
        win_gpu = vnp.array(np.ascontiguousarray(_window_view_1d(arr, window)), dtype=np.float32)
        mu = vnp.mean(win_gpu, axis=1, keepdims=True)
        diff = win_gpu - mu
        var = vnp.mean(diff * diff, axis=1)
        return vnp.sqrt(vnp.maximum(var, 0.0) + float(eps))
    if arr.ndim == 2:
        cols = [rolling_std(arr[:, c], window, eps=eps).to_numpy() for c in range(arr.shape[1])]
        return vnp.array(np.stack(cols, axis=1), dtype=np.float32)
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def rolling_min(x: Any, window: int) -> vnp.ndarray:
    """Causal GPU rolling minimum (same length as input)."""
    arr = _to_np_f32(x)
    if arr.ndim == 1:
        win_gpu = vnp.array(np.ascontiguousarray(_window_view_1d(arr, window)), dtype=np.float32)
        return vnp.min(win_gpu, axis=1)
    if arr.ndim == 2:
        cols = [rolling_min(arr[:, c], window).to_numpy() for c in range(arr.shape[1])]
        return vnp.array(np.stack(cols, axis=1), dtype=np.float32)
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def rolling_max(x: Any, window: int) -> vnp.ndarray:
    """Causal GPU rolling maximum (same length as input)."""
    arr = _to_np_f32(x)
    if arr.ndim == 1:
        win_gpu = vnp.array(np.ascontiguousarray(_window_view_1d(arr, window)), dtype=np.float32)
        return vnp.max(win_gpu, axis=1)
    if arr.ndim == 2:
        cols = [rolling_max(arr[:, c], window).to_numpy() for c in range(arr.shape[1])]
        return vnp.array(np.stack(cols, axis=1), dtype=np.float32)
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def ewma(x: Any, alpha: float = 0.2) -> vnp.ndarray:
    """Exponentially Weighted Moving Average returned in Vulkan GPU VRAM."""
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    arr = _to_np_f32(x)
    if arr.ndim == 1:
        out = np.empty_like(arr)
        out[0] = arr[0]
        one_m = 1.0 - a
        for i in range(1, len(arr)):
            out[i] = a * arr[i] + one_m * out[i - 1]
        return vnp.array(out, dtype=np.float32)
    if arr.ndim == 2:
        cols = [ewma(arr[:, c], alpha=a).to_numpy() for c in range(arr.shape[1])]
        return vnp.array(np.stack(cols, axis=1), dtype=np.float32)
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def make_lag_matrix(
    x: Any,
    lags: Union[int, Sequence[int]] = 12,
    *,
    horizon: int = 1,
    target_col: int = 0,
    include_rolling: Optional[Sequence[int]] = None,
) -> Tuple[vnp.ndarray, vnp.ndarray]:
    """Build supervised ``(X_gpu, y_gpu)`` forecasting matrices from a 1D or 2D time series.

    Parameters
    ----------
    x : array-like
        1D ``(T,)`` or 2D ``(T, C)`` chronological series.
    lags : int or Sequence[int]
        If an int ``L``, uses consecutive lags ``[1, 2, ..., L]`` (oldest to newest).
        If a sequence, uses those explicit positive lag offsets (e.g. ``[1, 2, 3, 24, 168]``).
    horizon : int
        Steps ahead to predict (default ``1`` = next step).
    target_col : int
        Column index to predict when ``x`` is 2D ``(T, C)``.
    include_rolling : Sequence[int], optional
        Optional causal rolling-mean window sizes appended as extra engineered features!
    """
    arr = _to_np_f32(x)
    if arr.ndim == 1:
        arr2 = arr[:, None]
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")

    t_len, n_cols = arr2.shape
    lag_list = list(range(1, int(lags) + 1)) if isinstance(lags, int) else [int(l) for l in lags]
    if not lag_list or any(l < 1 for l in lag_list):
        raise ValueError("lags must be positive integers")
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    max_lag = max(lag_list)
    start = max_lag - 1
    end = t_len - h
    if end <= start:
        raise ValueError(f"Series length {t_len} too short for max_lag={max_lag} and horizon={h}")

    n_samples = end - start
    feat_blocks = []
    # For sample at anchor index t (from start..end-1), target is t + h,
    # lag=1 is value at t, lag=k is value at t - (k - 1).
    for k in lag_list:
        offset = k - 1
        feat_blocks.append(arr2[start - offset : end - offset, :])

    if include_rolling:
        for w in include_rolling:
            rm = rolling_mean(arr2[:, target_col], int(w)).to_numpy()
            feat_blocks.append(rm[start:end, None])

    X_np = np.concatenate(feat_blocks, axis=1).astype(np.float32, copy=False)
    y_np = arr2[start + h : end + h, target_col].astype(np.float32, copy=False)
    assert X_np.shape[0] == n_samples and y_np.shape[0] == n_samples
    return vnp.array(np.ascontiguousarray(X_np), dtype=np.float32), vnp.array(np.ascontiguousarray(y_np), dtype=np.float32)


def sliding_windows(
    x: Any,
    window_size: int,
    *,
    stride: int = 1,
    channel_first: bool = False,
) -> vnp.ndarray:
    """Slice a 1D ``(T,)`` or 2D ``(T, C)`` time series into GPU windows for 1D CNNs / Transformers.

    Returns ``(N, W)`` for 1D input (or ``(N, 1, W)`` if ``channel_first=True``),
    and ``(N, W, C)`` for 2D input (or ``(N, C, W)`` if ``channel_first=True``).
    """
    arr = _to_np_f32(x)
    w = int(window_size)
    s = int(stride)
    if w < 1 or s < 1:
        raise ValueError("window_size and stride must be >= 1")
    if arr.shape[0] < w:
        raise ValueError(f"Series length {arr.shape[0]} < window_size {w}")

    if arr.ndim == 1:
        views = np.lib.stride_tricks.sliding_window_view(arr, w)[::s]
        if channel_first:
            views = views[:, None, :]
        return vnp.array(np.ascontiguousarray(views), dtype=np.float32)

    if arr.ndim == 2:
        # sliding_window_view over axis 0 gives (N, C, W)
        views = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)[::s]
        if not channel_first:
            views = np.transpose(views, (0, 2, 1))
        return vnp.array(np.ascontiguousarray(views), dtype=np.float32)

    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def temporal_split(
    x: Any,
    y: Optional[Any] = None,
    *,
    train_ratio: float = 0.8,
) -> Union[Tuple[vnp.ndarray, vnp.ndarray], Tuple[vnp.ndarray, vnp.ndarray, vnp.ndarray, vnp.ndarray]]:
    """Leak-free chronological train/test split (never shuffles future into past!)."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    x_np = _to_np_f32(x)
    split_idx = max(1, min(len(x_np) - 1, int(round(len(x_np) * train_ratio))))
    x_tr = vnp.array(np.ascontiguousarray(x_np[:split_idx]), dtype=np.float32)
    x_te = vnp.array(np.ascontiguousarray(x_np[split_idx:]), dtype=np.float32)
    if y is None:
        return x_tr, x_te
    y_np = _to_np_f32(y)
    if len(y_np) != len(x_np):
        raise ValueError(f"x length ({len(x_np)}) != y length ({len(y_np)})")
    y_tr = vnp.array(np.ascontiguousarray(y_np[:split_idx]), dtype=np.float32)
    y_te = vnp.array(np.ascontiguousarray(y_np[split_idx:]), dtype=np.float32)
    return x_tr, x_te, y_tr, y_te


def fourier_features(timestamps: Any, period: float, *, harmonics: int = 2) -> vnp.ndarray:
    """Compute GPU ``sin``/``cos`` Fourier harmonic features for a known cycle ``period``."""
    p = float(period)
    if p <= 0.0:
        raise ValueError(f"period must be > 0, got {period}")
    t_gpu = timestamps if isinstance(timestamps, vnp.ndarray) else vnp.array(_to_np_f32(timestamps).ravel(), dtype=np.float32)
    cols = []
    for k in range(1, int(harmonics) + 1):
        ang = t_gpu * np.float32(2.0 * np.pi * k / p)
        cols.append(vnp.sin(ang))
        cols.append(vnp.cos(ang))
    return vnp.column_stack(cols)
