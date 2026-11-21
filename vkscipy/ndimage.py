"""GPU 2D image filtering (scipy.ndimage compatible): convolve, gaussian_filter, sobel, laplace, uniform_filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import vknumpy as vnp

from .special import _to_f32


def _pad_edge_2d(img: vnp.ndarray, ph: int, pw: int) -> vnp.ndarray:
    h, w = img.shape
    # Pad using fast GPU slice/concat or numpy-free pad
    arr = img.to_numpy()
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="edge")
    return vnp.array(padded.astype(np.float32), dtype=np.float32)


def convolve(img: Any, kernel: Any) -> vnp.ndarray:
    """2D convolution with arbitrary (kh, kw) kernel on Vulkan GPU."""
    ig = _to_f32(img)
    if ig.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {ig.shape}")
    k_np = np.asarray(kernel, dtype=np.float32)[::-1, ::-1]
    kh, kw = k_np.shape
    ph, pw = kh // 2, kw // 2
    h, w = ig.shape
    pad = _pad_edge_2d(ig, ph, pw)
    out = vnp.zeros((h, w), dtype=np.float32)
    for dy in range(kh):
        for dx in range(kw):
            w_val = float(k_np[dy, dx])
            if abs(w_val) < 1e-12:
                continue
            out = out + pad[dy : dy + h, dx : dx + w] * w_val
    return out


def _convolve_separable_2d(img: Any, k1d: np.ndarray) -> vnp.ndarray:
    ig = _to_f32(img)
    h, w = ig.shape
    r = len(k1d) // 2
    # Horizontal pass
    pad_x = _pad_edge_2d(ig, 0, r)
    tmp = vnp.zeros((h, w), dtype=np.float32)
    for dx, wv in enumerate(k1d):
        if abs(float(wv)) > 1e-12:
            tmp = tmp + pad_x[:, dx : dx + w] * float(wv)
    # Vertical pass
    pad_y = _pad_edge_2d(tmp, r, 0)
    out = vnp.zeros((h, w), dtype=np.float32)
    for dy, wv in enumerate(k1d):
        if abs(float(wv)) > 1e-12:
            out = out + pad_y[dy : dy + h, :] * float(wv)
    return out


def gaussian_filter(img: Any, sigma: float = 1.0, truncate: float = 3.0) -> vnp.ndarray:
    """Separable 2D Gaussian blur on Vulkan GPU."""
    if sigma <= 0.0:
        return _to_f32(img)
    r = max(1, int(round(truncate * sigma)))
    xs = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (xs / float(sigma)) ** 2)
    k /= k.sum()
    return _convolve_separable_2d(img, k)


def uniform_filter(img: Any, size: int = 3) -> vnp.ndarray:
    """Separable 2D box/mean filter on Vulkan GPU."""
    s = max(1, int(size))
    if s % 2 == 0:
        s += 1
    k = np.full((s,), 1.0 / s, dtype=np.float32)
    return _convolve_separable_2d(img, k)


def sobel(img: Any, axis: Optional[int] = None) -> vnp.ndarray:
    """2D Sobel edge filter on Vulkan GPU (axis=0, axis=1, or magnitude when axis=None)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    if axis == 0:
        return convolve(img, ky)
    if axis == 1:
        return convolve(img, kx)
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    return vnp.sqrt(gx * gx + gy * gy)


def laplace(img: Any) -> vnp.ndarray:
    """2D discrete Laplacian filter on Vulkan GPU."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return convolve(img, k)
