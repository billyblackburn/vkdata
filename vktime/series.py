"""vktime.series: In-VRAM Vulkan GPU TimeSeries container with resampling, detrending, & persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import vknumpy as vnp

from .features import _to_np_f32, rolling_mean


class TimeSeries:
    """In-VRAM Vulkan GPU Time-Series container pairing timestamps with 1D or 2D ``(T, C)`` values.

    Examples
    --------
    >>> ts = TimeSeries(timestamps, values, name="sunspots")
    >>> detrended = ts.detrend(method="rolling", window=12)
    >>> downsampled = ts.resample(step=4, agg="mean")
    """

    def __init__(
        self,
        timestamps: Any,
        values: Any,
        *,
        name: str = "series",
        freq: Optional[str] = None,
    ) -> None:
        t_np = np.asarray(timestamps).ravel()
        v_np = _to_np_f32(values)
        if v_np.ndim not in (1, 2):
            raise ValueError(f"values must be 1D or 2D, got shape {v_np.shape}")
        if len(t_np) != v_np.shape[0]:
            raise ValueError(f"timestamps length ({len(t_np)}) != values rows ({v_np.shape[0]})")

        # Ensure chronological order
        if len(t_np) > 1 and np.any(t_np[1:] < t_np[:-1]):
            order = np.argsort(t_np)
            t_np = t_np[order]
            v_np = v_np[order]

        self.timestamps: np.ndarray = t_np
        self.values: vnp.ndarray = vnp.array(np.ascontiguousarray(v_np), dtype=np.float32)
        self.name: str = str(name)
        self.freq: Optional[str] = freq

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def shape(self) -> tuple:
        return tuple(self.values.shape)

    def to_numpy(self) -> np.ndarray:
        return self.values.to_numpy()

    def slice_time(self, start: Any = None, end: Any = None) -> "TimeSeries":
        """Filter the series to ``start <= timestamp <= end``."""
        mask = np.ones(len(self.timestamps), dtype=bool)
        if start is not None:
            mask &= self.timestamps >= start
        if end is not None:
            mask &= self.timestamps <= end
        return TimeSeries(
            self.timestamps[mask],
            self.values.to_numpy()[mask],
            name=self.name,
            freq=self.freq,
        )

    def resample(self, step: int, *, agg: str = "mean") -> "TimeSeries":
        """Downsample the time series into fixed buckets of ``step`` points on Vulkan GPU."""
        s = int(step)
        if s < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        n = len(self)
        n_bins = n // s
        if n_bins == 0:
            raise ValueError(f"Series length {n} < resample step {s}")

        arr = self.values.to_numpy()[: n_bins * s]
        ts_new = self.timestamps[: n_bins * s : s]
        agg_fn = agg.lower()
        if agg_fn not in ("mean", "sum", "min", "max"):
            raise ValueError(f"Unsupported agg {agg!r}; expected 'mean', 'sum', 'min', or 'max'")

        if arr.ndim == 1:
            binned = vnp.array(np.ascontiguousarray(arr.reshape(n_bins, s)), dtype=np.float32)
            out = getattr(vnp, agg_fn)(binned, axis=1)
        else:
            c = arr.shape[1]
            cols = []
            for ci in range(c):
                binned = vnp.array(np.ascontiguousarray(arr[:, ci].reshape(n_bins, s)), dtype=np.float32)
                cols.append(getattr(vnp, agg_fn)(binned, axis=1).to_numpy())
            out = vnp.array(np.stack(cols, axis=1), dtype=np.float32)
        return TimeSeries(ts_new, out, name=self.name, freq=self.freq)

    def diff(self, periods: int = 1) -> "TimeSeries":
        """First (or k-th) difference ``x[t] - x[t - periods]``."""
        p = int(periods)
        if p < 1 or p >= len(self):
            raise ValueError(f"periods must be in [1, {len(self) - 1}], got {periods}")
        arr = self.values.to_numpy()
        d = vnp.array(np.ascontiguousarray(arr[p:]), dtype=np.float32) - vnp.array(
            np.ascontiguousarray(arr[:-p]), dtype=np.float32
        )
        return TimeSeries(self.timestamps[p:], d, name=f"{self.name}_diff{p}", freq=self.freq)

    def pct_change(self, periods: int = 1, *, eps: float = 1e-8) -> "TimeSeries":
        """Percentage change ``(x[t] - x[t - periods]) / (|x[t - periods]| + eps)``."""
        p = int(periods)
        if p < 1 or p >= len(self):
            raise ValueError(f"periods must be in [1, {len(self) - 1}], got {periods}")
        arr = self.values.to_numpy()
        cur = vnp.array(np.ascontiguousarray(arr[p:]), dtype=np.float32)
        prev = vnp.array(np.ascontiguousarray(arr[:-p]), dtype=np.float32)
        pct = (cur - prev) / (vnp.abs(prev) + float(eps))
        return TimeSeries(self.timestamps[p:], pct, name=f"{self.name}_pct{p}", freq=self.freq)

    def detrend(self, *, method: str = "linear", window: int = 25) -> "TimeSeries":
        """Remove slow baseline drift (``'linear'`` least-squares or ``'rolling'`` moving average)."""
        m = method.lower()
        arr = self.values.to_numpy()
        if m == "rolling":
            trend = rolling_mean(self.values, window)
            return TimeSeries(self.timestamps, self.values - trend, name=f"{self.name}_detrend", freq=self.freq)
        if m == "linear":
            t_idx = np.arange(len(self), dtype=np.float64)
            if arr.ndim == 1:
                slope, intercept = np.polyfit(t_idx, arr.astype(np.float64), 1)
                trend = (slope * t_idx + intercept).astype(np.float32)
            else:
                trend = np.zeros_like(arr)
                for c in range(arr.shape[1]):
                    slope, intercept = np.polyfit(t_idx, arr[:, c].astype(np.float64), 1)
                    trend[:, c] = (slope * t_idx + intercept).astype(np.float32)
            return TimeSeries(
                self.timestamps,
                self.values - vnp.array(trend, dtype=np.float32),
                name=f"{self.name}_detrend",
                freq=self.freq,
            )
        raise ValueError(f"Unsupported detrend method {method!r}; expected 'linear' or 'rolling'")

    def normalize(self, *, eps: float = 1e-8) -> "TimeSeries":
        """Z-score normalize values in Vulkan GPU VRAM."""
        mu = vnp.mean(self.values, axis=0, keepdims=True) if self.values.ndim == 2 else vnp.mean(self.values)
        centered = self.values - mu
        var = vnp.mean(centered * centered, axis=0, keepdims=True) if self.values.ndim == 2 else vnp.mean(centered * centered)
        sd = vnp.sqrt(vnp.maximum(var, 0.0) + float(eps))
        return TimeSeries(self.timestamps, centered / sd, name=f"{self.name}_norm", freq=self.freq)

    def save(self, path: Union[str, Path]) -> Path:
        """Persist timestamps + values (.npz) and metadata (.meta.json) to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, timestamps=self.timestamps, values=self.values.to_numpy())
        p.with_suffix(".meta.json").write_text(
            json.dumps({"name": self.name, "freq": self.freq}),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TimeSeries":
        """Reload a saved ``TimeSeries`` directly into Vulkan GPU VRAM."""
        p = Path(path)
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        data = np.load(p)
        return cls(data["timestamps"], data["values"], name=meta["name"], freq=meta["freq"])
