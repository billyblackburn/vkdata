"""vktime: Pure-Vulkan GPU Time-Series Database, Rolling Windows, Spectral Analysis & Forecasting."""

from .features import (
    ewma,
    fourier_features,
    make_lag_matrix,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    sliding_windows,
    temporal_split,
)
from .series import TimeSeries, read_csv
from .spectral import (
    autocorrelation,
    detect_anomalies,
    periodogram,
    phase_fold,
    seasonal_decompose,
)

__all__ = [
    "TimeSeries",
    "autocorrelation",
    "detect_anomalies",
    "ewma",
    "fourier_features",
    "make_lag_matrix",
    "periodogram",
    "phase_fold",
    "read_csv",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_std",
    "seasonal_decompose",
    "sliding_windows",
    "temporal_split",
]
