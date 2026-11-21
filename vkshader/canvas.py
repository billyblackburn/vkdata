"""GPU Canvas, SpatialPoints index, and GPU shading for vkshader."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
import vknumpy as vnp

from ._backend import raster_lines_gpu, shade_aux_rgba_gpu, shade_rgba_gpu
from .basemap import get_esri_mosaic_gpu

_CAT_CODES_CACHE: dict = {}


_NAMED_COLORS = {
    "red": (255, 65, 54),
    "blue": (60, 145, 255),
    "green": (46, 204, 64),
    "orange": (255, 133, 27),
    "yellow": (255, 220, 0),
    "purple": (177, 13, 201),
    "teal": (57, 204, 204),
    "cyan": (0, 220, 255),
    "magenta": (255, 65, 180),
    "white": (255, 255, 255),
}


def _parse_color_u32(c: Any) -> int:
    if isinstance(c, str):
        s = c.strip().lower()
        if s in _NAMED_COLORS:
            r, g, b = _NAMED_COLORS[s]
        elif s.startswith("#") and len(s) == 7:
            r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
        else:
            r, g, b = 255, 180, 50
    else:
        r, g, b = int(c[0]), int(c[1]), int(c[2])
    return (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | (0xFF << 24)


def _hsv_palette_u32(n: int = 256, reverse: bool = True) -> np.ndarray:
    """Build a 256-entry packed RGBA uint32 HSV palette (matching matplotlib hsv_r)."""
    out = np.zeros(256, dtype=np.uint32)
    m = max(1, min(int(n), 256))
    for i in range(256):
        h = (1.0 - (i % m) / float(m)) if reverse else ((i % m) / float(m))
        h6 = (h % 1.0) * 6.0
        k = int(h6) % 6
        f = h6 - int(h6)
        if k == 0:   rgb = (1.0, f, 0.0)
        elif k == 1: rgb = (1.0 - f, 1.0, 0.0)
        elif k == 2: rgb = (0.0, 1.0, f)
        elif k == 3: rgb = (0.0, 1.0 - f, 1.0)
        elif k == 4: rgb = (f, 0.0, 1.0)
        else:        rgb = (1.0, 0.0, 1.0 - f)
        r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        out[i] = r | (g << 8) | (b << 16) | (0xFF << 24)
    return out


def _build_palette_gpu(color_key: Any, n_cats: int = 256) -> vnp.ndarray:
    if color_key is None or color_key == "hsv_r":
        pal = _hsv_palette_u32(n_cats, reverse=True)
    elif color_key == "hsv":
        pal = _hsv_palette_u32(n_cats, reverse=False)
    else:
        pal = np.zeros(256, dtype=np.uint32)
        colors = list(color_key)
        for i in range(256):
            pal[i] = _parse_color_u32(colors[i % len(colors)])
    return vnp.array(pal.view(np.int32), dtype=np.int32)


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
    """2D GPU aggregation grid produced by Canvas.points() or Canvas.line()."""

    grid: vnp.ndarray
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    aux: Optional[vnp.ndarray] = None
    mode: str = "count"
    vmin: float = 0.0
    vmax: float = 250.0


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

    def line(
        self,
        source: Any,
        x: str,
        y: str,
        agg: str = "count",
        column: Optional[str] = None,
        color_key: Any = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> AggResult:
        """Rasterize NaN-separated polylines (trajectories) on Vulkan GPU.

        Supports ``agg='count'``, ``agg='mean'`` (with ``column``), and
        ``agg='by'`` (categorical coloring with ``column`` and optional ``color_key``).
        """
        xr = x_range or self.x_range
        yr = y_range or self.y_range
        xs = _extract_gpu_col(source, x)
        if xs.dtype != np.float32:
            xs = xs.astype(np.float32)
        ys = _extract_gpu_col(source, y)
        if ys.dtype != np.float32:
            ys = ys.astype(np.float32)
        if xr is None:
            xr = (float(vnp.nanmin(xs).item()), float(vnp.nanmax(xs).item()))
        if yr is None:
            yr = (float(vnp.nanmin(ys).item()), float(vnp.nanmax(ys).item()))

        if agg == "count":
            grid, _ = raster_lines_gpu(xs, ys, self.plot_width, self.plot_height, xr, yr, mode=0)
            return AggResult(grid=grid, x_range=(float(xr[0]), float(xr[1])), y_range=(float(yr[0]), float(yr[1])), mode="count")

        if column is None:
            raise ValueError(f"column is required for agg={agg!r}")

        if agg == "mean":
            vals = _extract_gpu_col(source, column)
            if vals.dtype != np.float32:
                vals = vals.astype(np.float32)
            grid, aux = raster_lines_gpu(xs, ys, self.plot_width, self.plot_height, xr, yr, mode=1, vals=vals)
            lo = float(vmin) if vmin is not None else float(vnp.nanmin(vals).item())
            hi = float(vmax) if vmax is not None else float(vnp.nanmax(vals).item())
            return AggResult(
                grid=grid, aux=aux,
                x_range=(float(xr[0]), float(xr[1])), y_range=(float(yr[0]), float(yr[1])),
                mode="mean", vmin=lo, vmax=hi,
            )

        if agg == "by":
            col_obj = source[column]
            cache_key = (id(source), str(column))
            cached_cat = _CAT_CODES_CACHE.get(cache_key)
            if cached_cat is not None and cached_cat[0] is col_obj:
                codes, n_cats = cached_cat[1], cached_cat[2]
            else:
                if hasattr(col_obj, "values") and isinstance(col_obj.values, vnp.ndarray):
                    codes = col_obj.values if col_obj.values.dtype == np.int32 else col_obj.values.astype(np.int32)
                    n_cats = max(2, int(vnp.max(codes).item()) + 1)
                elif hasattr(col_obj, "_codes") and isinstance(col_obj._codes, vnp.ndarray):
                    codes = col_obj._codes if col_obj._codes.dtype == np.int32 else col_obj._codes.astype(np.int32)
                    n_cats = max(2, len(getattr(col_obj, "_categories", [])) or int(vnp.max(codes).item()) + 1)
                else:
                    try:
                        import pyarrow as pa
                        import pyarrow.compute as pc
                        raw = col_obj.values if hasattr(col_obj, "values") else col_obj
                        enc = pc.dictionary_encode(pa.array(raw))
                        idx_np = enc.indices.to_numpy(zero_copy_only=False).astype(np.int32)
                        codes = vnp.array(idx_np, dtype=np.int32)
                        n_cats = max(2, len(enc.dictionary))
                    except Exception:
                        arr = np.asarray(col_obj)
                        if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
                            codes = vnp.array(arr.astype(np.int32), dtype=np.int32)
                            n_cats = max(2, int(arr.max()) + 1) if arr.size else 2
                        else:
                            _, inv = np.unique(arr, return_inverse=True)
                            codes = vnp.array(inv.astype(np.int32), dtype=np.int32)
                            n_cats = max(2, int(inv.max()) + 1) if inv.size else 2
                if len(_CAT_CODES_CACHE) > 16:
                    _CAT_CODES_CACHE.clear()
                _CAT_CODES_CACHE[cache_key] = (col_obj, codes, n_cats)

            pal_gpu = _build_palette_gpu(color_key, n_cats=n_cats)
            grid, aux = raster_lines_gpu(
                xs, ys, self.plot_width, self.plot_height, xr, yr,
                mode=2, vals=codes, palette=pal_gpu,
            )
            max_c = float(vnp.max(grid).item()) if grid.size else 1.0
            return AggResult(
                grid=grid, aux=aux,
                x_range=(float(xr[0]), float(xr[1])), y_range=(float(yr[0]), float(yr[1])),
                mode="cat", vmin=0.0, vmax=max(max_c, 1.0),
            )

        raise ValueError(f"Unsupported agg={agg!r}; expected 'count', 'mean', or 'by'")


def shade(
    agg: AggResult,
    how: str = "eq_hist",
    alpha: int = 235,
    basemap: Optional[str] = None,
    zoom: Optional[int] = None,
    spread_px: int = -1,
    cmap: str = "fire",
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
    if agg.mode == "count":
        rgba_gpu = shade_rgba_gpu(
            agg.grid, how=how, alpha=float(alpha) / 255.0,
            bg=bg_gpu, uv_rect=uv_rect, spread_px=spread_px,
        )
    elif agg.mode == "mean":
        assert agg.aux is not None
        m = 2 if cmap == "fire_r" else 1
        rgba_gpu = shade_aux_rgba_gpu(
            agg.grid, agg.aux, mode=m, alpha=float(alpha) / 255.0,
            vmin=agg.vmin, vmax=agg.vmax, bg=bg_gpu, uv_rect=uv_rect,
        )
    elif agg.mode == "cat":
        assert agg.aux is not None
        rgba_gpu = shade_aux_rgba_gpu(
            agg.grid, agg.aux, mode=3, alpha=float(alpha) / 255.0,
            vmin=agg.vmin, vmax=agg.vmax, bg=bg_gpu, uv_rect=uv_rect,
        )
    else:
        raise ValueError(f"Unsupported agg.mode={agg.mode!r}")
    rgba = rgba_gpu.to_numpy().view(np.uint8).reshape(h, w, 4)
    return ShadedImage(rgba=rgba, x_range=agg.x_range, y_range=agg.y_range)
