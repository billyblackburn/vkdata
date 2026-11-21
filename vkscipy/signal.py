"""GPU 1D signal processing (scipy.signal compatible): convolve, correlate, spectrogram."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import vknumpy as vnp

from .special import _to_f32


def convolve(a: Any, v: Any, mode: str = "same") -> vnp.ndarray:
    """1D convolution on Vulkan GPU ('same', 'valid', or 'full')."""
    ag = _to_f32(a).ravel()
    k_np = np.asarray(v, dtype=np.float32).ravel()[::-1]
    n = ag.size
    m = len(k_np)
    if mode == "full":
        pad_l, pad_r, out_n = m - 1, m - 1, n + m - 1
    elif mode == "same":
        pad_l = (m - 1) // 2
        pad_r = (m - 1) - pad_l
        out_n = n
    elif mode == "valid":
        if n < m:
            return vnp.zeros((0,), dtype=np.float32)
        pad_l, pad_r, out_n = 0, 0, n - m + 1
    else:
        raise ValueError(f"Unsupported mode={mode!r}")

    if pad_l > 0 or pad_r > 0:
        padded = vnp.array(np.pad(ag.to_numpy(), (pad_l, pad_r), mode="constant").astype(np.float32), dtype=np.float32)
    else:
        padded = ag

    out = vnp.zeros((out_n,), dtype=np.float32)
    for i, wv in enumerate(k_np):
        if abs(float(wv)) > 1e-12:
            out = out + padded[i : i + out_n] * float(wv)
    return out


def correlate(a: Any, v: Any, mode: str = "same") -> vnp.ndarray:
    """1D cross-correlation on Vulkan GPU."""
    k_rev = np.asarray(v, dtype=np.float32).ravel()[::-1]
    return convolve(a, k_rev, mode=mode)


def spectrogram(
    x: Any,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: int = 128,
) -> Tuple[vnp.ndarray, vnp.ndarray, vnp.ndarray]:
    """Compute STFT power spectrogram via batched Vulkan GPU GEMM (Hann-windowed DFT).

    Returns ``(freqs, times, Sxx)`` as GPU ``vknumpy.ndarray``s.
    """
    x_np = _to_f32(x).ravel().to_numpy()
    n = len(x_np)
    step = max(1, int(nperseg) - int(noverlap))
    n_frames = max(1, (n - nperseg) // step + 1)

    # Frame signal & apply Hann window
    frames = np.zeros((n_frames, nperseg), dtype=np.float32)
    win = np.hanning(nperseg).astype(np.float32)
    for i in range(n_frames):
        s = i * step
        chunk = x_np[s : s + nperseg]
        frames[i, : len(chunk)] = chunk * win[: len(chunk)]

    n_freq = nperseg // 2 + 1
    k = np.arange(n_freq, dtype=np.float32)[:, None]
    t_idx = np.arange(nperseg, dtype=np.float32)[None, :]
    ang = (-2.0 * np.pi / nperseg) * (k @ t_idx)
    W_cos = vnp.array(np.cos(ang).T.astype(np.float32), dtype=np.float32)  # (nperseg, n_freq)
    W_sin = vnp.array(np.sin(ang).T.astype(np.float32), dtype=np.float32)

    F_g = vnp.array(frames, dtype=np.float32)
    re = F_g @ W_cos
    im = F_g @ W_sin
    scale = float(1.0 / (fs * np.sum(win * win)))
    Sxx = ((re * re + im * im) * scale).T  # (n_freq, n_frames)

    freqs = vnp.array(np.linspace(0.0, fs * 0.5, n_freq, dtype=np.float32), dtype=np.float32)
    times = vnp.array((np.arange(n_frames, dtype=np.float32) * step + nperseg * 0.5) / fs, dtype=np.float32)
    return freqs, times, Sxx
