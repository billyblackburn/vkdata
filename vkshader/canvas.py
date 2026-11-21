"""GPU Canvas, SpatialPoints index, and GPU shading for vkshader."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
import vknumpy as vnp

from ._backend import shade_rgba_gpu
from .basemap import get_esri_mosaic_gpu


def _extract_gpu_col(source: Any, col: str) -> vnp.ndarray:
    c = source[col]
    if hasattr(c, "values") and isinstance(c.values, vnp.ndarray):
        return c.values
    if isinstance(c, vnp.ndarray):
        return c
    return vnp.asarray(c, dtype=vnp.float32)


class SpatialPoints:
    """Pre-sorted GPU (x, y) columns with a tiny coarse x-index for fast pan/zoom slicing."""

    def __init__(
        self,
        x: vnp.ndarray,
        y: vnp.ndarray,
        extra: Optional[dict] = None,
        num_buckets: int = 2048,
    ) -> None:
        xf = vnp.asarray(x, dtype=vnp.float32).ravel()
        yf = vnp.asarray(y, dtype=vnp.float32).ravel()
        order = vnp.argsort(xf)
        self.x = vnp.take(xf, order)
        self.y = vnp.take(yf, order)
        self.extra: dict = {}
        if extra:
            for col_name, col_arr in extra.items():
                cf = vnp.asarray(col_arr, dtype=vnp.float32).ravel()
                self.extra[col_name] = vnp.take(cf, order)
        self.n = self.x.size
        k = min(num_buckets, max(2, self.n))
        self._sample_idx = np.linspace(0, max(0, self.n - 1), k, dtype=np.int32)
        self._sample_x = vnp.take(self.x, vnp.array(self._sample_idx, dtype=vnp.int32)).to_numpy()

    def slice_bounds(self, xmin: float, xmax: float) -> Tuple[int, int]:
        if self.n == 0:
            return 0, 0
        b_lo = int(np.searchsorted(self._sample_x, xmin, side="left"))
        b_hi = int(np.searchsorted(self._sample_x, xmax, side="right"))
        lo = int(self._sample_idx[max(0, b_lo - 1)])
        hi = int(self._sample_idx[min(len(self._sample_idx) - 1, b_hi)]) + 1
        return lo, hi

    def slice_x(self, xmin: float, xmax: float) -> Tuple[vnp.ndarray, vnp.ndarray]:
        lo, hi = self.slice_bounds(xmin, xmax)
        return self.x[lo:hi], self.y[lo:hi]

    def slice_extra(self, name: str, xmin: float, xmax: float) -> vnp.ndarray:
        lo, hi = self.slice_bounds(xmin, xmax)
        return self.extra[name][lo:hi]


@dataclass
class AggResult:
    """2D GPU count grid produced by Canvas.points()."""

    grid: vnp.ndarray
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


@dataclass
class ShadedImage:
    """RGBA uint8 image shaded & composited on GPU."""

    rgba: np.ndarray
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self.rgba, mode="RGBA")

    def to_jpeg(self, quality: int = 85) -> bytes:
        return encode_jpeg(self.rgba, quality=quality)

    def save(self, path: str | Path, **kwargs: Any) -> None:
        p = Path(path)
        if p.suffix.lower() in (".jpg", ".jpeg"):
            Image.fromarray(self.rgba[:, :, :3], mode="RGB").save(p, format="JPEG", **kwargs)
        else:
            self.to_pil().save(p, **kwargs)

    def _repr_png_(self) -> bytes:
        buf = io.BytesIO()
        self.to_pil().save(buf, format="PNG", compress_level=1)
        return buf.getvalue()


def encode_jpeg(rgba: np.ndarray, quality: int = 85) -> bytes:
    """Encode an (H, W, 4) RGBA array to JPEG bytes (~5-9 ms at 720x720)."""
    buf = io.BytesIO()
    Image.fromarray(rgba[:, :, :3], mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class Canvas:
    """GPU aggregation canvas, analogous to datashader.Canvas."""

    def __init__(
        self,
        plot_width: int = 800,
        plot_height: int = 800,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.plot_width = int(plot_width)
        self.plot_height = int(plot_height)
        self.x_range = x_range
        self.y_range = y_range

    def points(
        self,
        source: Any,
        x: Optional[str] = None,
        y: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> AggResult:
        """Aggregate 2D points onto the canvas in a single zero-copy Vulkan atomicAdd pass."""
        xr = x_range or self.x_range
        yr = y_range or self.y_range

        if isinstance(source, SpatialPoints):
            if xr is None or yr is None:
                raise ValueError("x_range and y_range must be set on Canvas or points()")
            xs, ys = source.slice_x(xr[0], xr[1])
        else:
            if x is None or y is None:
                raise ValueError("x and y column names are required for DataFrame/dict sources")
            xs = _extract_gpu_col(source, x)
            ys = _extract_gpu_col(source, y)
            if xr is None:
                xr = (float(vnp.nanmin(xs).item()), float(vnp.nanmax(xs).item()))
            if yr is None:
                yr = (float(vnp.nanmin(ys).item()), float(vnp.nanmax(ys).item()))

        grid, _, _ = vnp.histogram2d(
            xs,
            ys,
            bins=(self.plot_height, self.plot_width),
            range=(xr, yr),
            flip_y=True,
        )
        return AggResult(grid=grid, x_range=(float(xr[0]), float(xr[1])), y_range=(float(yr[0]), float(yr[1])))


def shade(
    agg: AggResult,
    how: str = "eq_hist",
    alpha: int = 235,
    basemap: Optional[str] = None,
    zoom: Optional[int] = None,
    spread_px: int = -1,
) -> ShadedImage:
    """Shade a GPU AggResult (and optionally composite over an Esri basemap) 100% on GPU.

    ``spread_px=-1`` enables GPU automatic ``dynspread`` (dilates sparse zoomed-in pixels
    in VRAM with zero CPU round-trip). Set ``spread_px=0`` to disable or ``1..3`` for fixed radius.
    """
    h, w = int(agg.grid.shape[0]), int(agg.grid.shape[1])
    bg_gpu = None
    uv_rect = (0.0, 0.0, 1.0, 1.0)
    if basemap == "esri":
        res = get_esri_mosaic_gpu(agg.x_range, agg.y_range, w, h, zoom=zoom)
        if res is not None:
            bg_gpu, uv_rect = res
    rgba_gpu = shade_rgba_gpu(
        agg.grid, how=how, alpha=float(alpha) / 255.0,
        bg=bg_gpu, uv_rect=uv_rect, spread_px=spread_px,
    )
    rgba = rgba_gpu.to_numpy().view(np.uint8).reshape(h, w, 4)
    return ShadedImage(rgba=rgba, x_range=agg.x_range, y_range=agg.y_range)
