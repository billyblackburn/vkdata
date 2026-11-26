"""vktime.spectral: GPU autocorrelation, periodogram, anomaly detection, seasonal decomposition, & phase folding."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import vknumpy as vnp

from .features import _to_np_f32, rolling_mean, rolling_std


def autocorrelation(x: Any, max_lag: int = 50) -> vnp.ndarray:
    """Compute the Autocorrelation Function (ACF) for lags ``0..max_lag`` on Vulkan GPU."""
    arr = _to_np_f32(x).ravel()
    n = len(arr)
    ml = min(int(max_lag), n - 1)
    if ml < 0:
        raise ValueError("max_lag must be >= 0")

    centered = arr - float(np.mean(arr))
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        out = np.zeros(ml + 1, dtype=np.float32)
        out[0] = 1.0
        return vnp.array(out, dtype=np.float32)

    # Build lagged matrix (ml+1, n) and compute all lags in ONE Vulkan GPU matvec!
    lag_mat = np.zeros((ml + 1, n), dtype=np.float32)
    for lag in range(ml + 1):
        lag_mat[lag, : n - lag] = centered[lag:]

    acf_gpu = (vnp.array(lag_mat, dtype=np.float32) @ vnp.array(centered[:, None], dtype=np.float32)).reshape(-1)
    return acf_gpu / np.float32(denom)


def periodogram(x: Any, *, sample_rate: float = 1.0, detrend: bool = True) -> Tuple[np.ndarray, vnp.ndarray]:
    """Compute the one-sided Power Spectral Density ``(frequencies, power)`` of a 1D signal.

    Peaks in ``power`` reveal hidden periodicities (e.g. 11-year solar cycle,
    24-hour daily demand cycle, or an exoplanet's orbital period).
    """
    arr = _to_np_f32(x).ravel().astype(np.float64)
    if detrend:
        arr = arr - np.mean(arr)
    n = len(arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    spec = np.fft.rfft(arr)
    power = (np.abs(spec) ** 2) / max(n, 1)
    return freqs.astype(np.float32), vnp.array(power.astype(np.float32), dtype=np.float32)


def detect_anomalies(
    x: Any,
    *,
    window: int = 25,
    threshold: float = 3.0,
    direction: str = "both",
    robust: bool = True,
) -> Tuple[vnp.ndarray, np.ndarray, np.ndarray]:
    """Detect spikes or transit dips using local rolling Z-scores (robust MAD or mean/std).

    Parameters
    ----------
    x : array-like
        1D time series.
    window : int
        Causal rolling window size for local baseline & spread.
    threshold : float
        Z-score magnitude threshold (default ``3.0`` sigma).
    direction : {'both', 'high', 'low'}
        ``'both'`` flags spikes and dips; ``'low'`` flags downward dips (e.g. exoplanet transits!);
        ``'high'`` flags upward spikes.
    robust : bool
        If True (default), uses rolling median & MAD (``1.4826 * median(|x - med|)``) so
        multi-point spikes/dips cannot inflate their own standard deviation and mask themselves.

    Returns
    -------
    z_scores_gpu : vnp.ndarray
        Local rolling Z-score at every timestep.
    indices : np.ndarray
        Integer indices where anomalies were detected.
    mask : np.ndarray
        Boolean mask of detected anomalies.
    """
    arr_np = _to_np_f32(x).ravel()
    arr_gpu = vnp.array(arr_np, dtype=np.float32)
    if robust:
        from .features import _window_view_1d

        win = _window_view_1d(arr_np, window)
        med = np.median(win, axis=1).astype(np.float32)
        mad = np.median(np.abs(win - med[:, None]), axis=1).astype(np.float32)
        scale = np.maximum(1.4826 * mad, 1e-6).astype(np.float32)
        z_gpu = (arr_gpu - vnp.array(med, dtype=np.float32)) / vnp.array(scale, dtype=np.float32)
    else:
        mu = rolling_mean(arr_gpu, window)
        sd = rolling_std(arr_gpu, window, eps=1e-6)
        z_gpu = (arr_gpu - mu) / sd
    z_np = z_gpu.to_numpy()

    d = direction.lower()
    t = float(threshold)
    if d == "both":
        mask = np.abs(z_np) >= t
    elif d == "high":
        mask = z_np >= t
    elif d == "low":
        mask = z_np <= -t
    else:
        raise ValueError(f"direction must be 'both', 'high', or 'low', got {direction!r}")

    indices = np.nonzero(mask)[0].astype(np.int32)
    return z_gpu, indices, mask


def seasonal_decompose(x: Any, period: int) -> Tuple[vnp.ndarray, vnp.ndarray, vnp.ndarray]:
    """Additive seasonal decomposition into ``(trend, seasonal, residual)`` on Vulkan GPU."""
    p = int(period)
    if p < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    arr = _to_np_f32(x).ravel()
    n = len(arr)
    if n < 2 * p:
        raise ValueError(f"Series length {n} must be >= 2 * period ({2 * p})")

    # Centered moving average for smooth trend
    half = p // 2
    kernel = np.ones(p, dtype=np.float32) / float(p)
    pad = np.pad(arr, (half, p - 1 - half), mode="edge")
    trend = np.convolve(pad, kernel, mode="valid").astype(np.float32)

    detrended = arr - trend
    seasonal_pattern = np.array([np.mean(detrended[i::p]) for i in range(p)], dtype=np.float32)
    seasonal_pattern -= float(np.mean(seasonal_pattern))
    seasonal = np.resize(seasonal_pattern, n).astype(np.float32)
    residual = (arr - trend - seasonal).astype(np.float32)

    return (
        vnp.array(trend, dtype=np.float32),
        vnp.array(seasonal, dtype=np.float32),
        vnp.array(residual, dtype=np.float32),
    )


def phase_fold(
    times: Any,
    values: Any,
    period: float,
    *,
    t0: float = 0.0,
) -> Tuple[np.ndarray, vnp.ndarray]:
    """Fold a periodic signal (e.g. a star's light curve) onto phase ``[-0.5, 0.5)``.

    Repeating dips (like an exoplanet transiting every ``period`` days) align at
    ``phase = 0.0`` so even faint signals stand out unmistakably.
    """
    p = float(period)
    if p <= 0.0:
        raise ValueError(f"period must be > 0, got {period}")
    t_np = np.asarray(times, dtype=np.float64).ravel()
    v_np = _to_np_f32(values).ravel()
    if len(t_np) != len(v_np):
        raise ValueError("times and values must have the same length")

    phase = ((t_np - float(t0) + 0.5 * p) % p) / p - 0.5
    order = np.argsort(phase)
    return phase[order].astype(np.float32), v_np[order].astype(np.float32)
