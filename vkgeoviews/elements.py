"""CRS-aware geographic Elements (Points, Path, Polygons) with automatic Vulkan GPU projection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import vknumpy as vnp
import vkpandas as vpd
import vkspatial as vksp
from vkholoviews.core import Element
from vkholoviews.elements import Path as _HVPath, Points as _HVPoints


def _is_lonlat(crs: Optional[str]) -> bool:
    if crs is None:
        return False
    return crs.upper() in ("EPSG:4326", "4326", "WGS84", "LONLAT", "PLATECARREE")


def _project_source_to_mercator(data: Any, x_col: str, y_col: str) -> Any:
    if isinstance(data, vksp.GeoDataFrame):
        if _is_lonlat(data.crs):
            return data.to_crs("EPSG:3857").df
        return data.df
    if isinstance(data, vpd.DataFrame):
        out = data.copy()
        mx, my = vksp.lonlat_to_mercator(out[x_col], out[y_col])
        out[x_col] = mx
        out[y_col] = my
        return out
    if isinstance(data, dict):
        out_d = dict(data)
        mx, my = vksp.lonlat_to_mercator(out_d[x_col], out_d[y_col])
        out_d[x_col] = mx
        out_d[y_col] = my
        return out_d
    return data


def Points(
    data: Any,
    kdims: Optional[Sequence[str]] = None,
    vdims: Optional[Sequence[str]] = None,
    crs: Optional[str] = "EPSG:4326",
    **opts: Any,
) -> _HVPoints:
    """GeoViews-style Points Element that auto-projects EPSG:4326 -> Web Mercator on Vulkan GPU."""
    if isinstance(data, vksp.GeoDataFrame):
        kdims = kdims or [data.x_col, data.y_col]
        eff_crs = data.crs if data.crs is not None else crs
    else:
        if kdims is None:
            cols = list(getattr(data, "columns", []))
            kdims = cols[:2] if len(cols) >= 2 else ["lon", "lat"]
        eff_crs = crs

    proj_data = _project_source_to_mercator(data, kdims[0], kdims[1]) if _is_lonlat(eff_crs) else (
        data.df if isinstance(data, vksp.GeoDataFrame) else data
    )
    return _HVPoints(proj_data, kdims=kdims, vdims=vdims, **opts)


def Path(
    data: Any,
    kdims: Optional[Sequence[str]] = None,
    vdims: Optional[Sequence[str]] = None,
    crs: Optional[str] = "EPSG:4326",
    **opts: Any,
) -> _HVPath:
    """GeoViews-style Path Element that auto-projects EPSG:4326 -> Web Mercator on Vulkan GPU."""
    if isinstance(data, vksp.GeoDataFrame):
        kdims = kdims or [data.x_col, data.y_col]
        eff_crs = data.crs if data.crs is not None else crs
    else:
        if kdims is None:
            cols = list(getattr(data, "columns", []))
            kdims = cols[:2] if len(cols) >= 2 else ["lon", "lat"]
        eff_crs = crs

    proj_data = _project_source_to_mercator(data, kdims[0], kdims[1]) if _is_lonlat(eff_crs) else (
        data.df if isinstance(data, vksp.GeoDataFrame) else data
    )
    return _HVPath(proj_data, kdims=kdims, vdims=vdims, **opts)


def _fire_rgb(s: float) -> Tuple[int, int, int]:
    s = max(0.0, min(1.0, float(s)))
    stops = [
        (0.00, (35, 25, 90)),
        (0.25, (140, 25, 85)),
        (0.50, (220, 75, 20)),
        (0.75, (255, 175, 25)),
        (1.00, (255, 245, 165)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if s <= t1:
            u = (s - t0) / max(t1 - t0, 1e-6)
            return (
                int(c0[0] + u * (c1[0] - c0[0])),
                int(c0[1] + u * (c1[1] - c0[1])),
                int(c0[2] + u * (c1[2] - c0[2])),
            )
    return (255, 245, 165)


class Polygons(Element):
    """GeoViews-style Polygons & Choropleth Element (auto-projects EPSG:4326 -> EPSG:3857 on GPU)."""

    def __init__(
        self,
        polygons: Union[Dict[str, Any], Sequence[Any]],
        values: Optional[Union[Dict[str, float], Sequence[float]]] = None,
        crs: Optional[str] = "EPSG:4326",
        cmap: Any = "fire",
        alpha: int = 165,
        line_color: Tuple[int, int, int, int] = (255, 255, 255, 225),
        show_labels: bool = True,
        value_label: str = "",
        **opts: Any,
    ) -> None:
        super().__init__(data=polygons, opts=opts)
        self.crs = crs
        self.cmap = cmap
        self.alpha = int(alpha)
        self.line_color = line_color
        self.show_labels = bool(show_labels)
        self.value_label = value_label

        if isinstance(polygons, dict):
            names = list(polygons.keys())
            raw_rings = list(polygons.values())
        else:
            raw_rings = list(polygons)
            names = [f"poly_{i}" for i in range(len(raw_rings))]

        proj_rings: List[np.ndarray] = []
        for ring in raw_rings:
            arr = np.asarray(ring, dtype=np.float32)[:, :2]
            if _is_lonlat(crs) and len(arr) > 0:
                mx, my = vksp.lonlat_to_mercator(arr[:, 0], arr[:, 1])
                arr = np.column_stack([mx.to_numpy(), my.to_numpy()]).astype(np.float32)
            proj_rings.append(arr)

        self.names = names
        self.rings = proj_rings

        if values is None:
            self.values: Optional[List[float]] = None
        elif isinstance(values, dict):
            self.values = [float(values.get(nm, 0.0)) for nm in names]
        else:
            self.values = [float(v) for v in values]

    def bounds_mercator(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if not self.rings:
            return (-1.0, 1.0), (-1.0, 1.0)
        xs = np.concatenate([r[:, 0] for r in self.rings if len(r) > 0])
        ys = np.concatenate([r[:, 1] for r in self.rings if len(r) > 0])
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        px = max((x1 - x0) * 0.12, 100.0)
        py = max((y1 - y0) * 0.12, 100.0)
        return (x0 - px, x1 + px), (y0 - py, y1 + py)

    def draw_onto(
        self,
        base_img: Image.Image,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ) -> Image.Image:
        """Composite filled/outlined polygons & optional choropleth labels onto base_img."""
        w, h = base_img.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        inv_dx = 1.0 / max(x_range[1] - x_range[0], 1e-9)
        inv_dy = 1.0 / max(y_range[1] - y_range[0], 1e-9)

        vmin, vmax = 0.0, 1.0
        if self.values:
            vmin = min(self.values)
            vmax = max(self.values)
            if vmax <= vmin:
                vmax = vmin + 1.0

        for idx, (name, ring) in enumerate(zip(self.names, self.rings)):
            if len(ring) < 3:
                continue
            pts = [
                (
                    (float(x) - x_range[0]) * inv_dx * w,
                    (y_range[1] - float(y)) * inv_dy * h,
                )
                for x, y in ring
            ]
            if self.values is not None:
                from vkcolorcet import get_rgb_lut
                s = max(0.0, min(1.0, (self.values[idx] - vmin) / (vmax - vmin)))
                lut = get_rgb_lut(self.cmap)
                rgb = lut[int(round(s * 255.0))]
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            else:
                r, g, b = (65, 165, 255)
            draw.polygon(pts, fill=(r, g, b, self.alpha), outline=self.line_color)
            # Crisp 2px border
            draw.line(pts + [pts[0]], fill=self.line_color, width=2)

            if self.show_labels:
                cx, cy = vksp.polygon_centroid(ring)
                lx = (cx - x_range[0]) * inv_dx * w
                ly = (y_range[1] - cy) * inv_dy * h
                lbl = f"{name}\n{self.values[idx]:.2f}" if self.values is not None else name
                if 10 < lx < w - 10 and 10 < ly < h - 10:
                    tw = max(len(line) for line in lbl.split("\n")) * 6 + 8
                    th = 14 * len(lbl.split("\n")) + 4
                    draw.rectangle([lx - tw / 2, ly - th / 2, lx + tw / 2, ly + th / 2], fill=(12, 16, 24, 195))
                    draw.multiline_text((lx - tw / 2 + 4, ly - th / 2 + 2), lbl, fill=(240, 244, 250, 255))

        return Image.alpha_composite(base_img.convert("RGBA"), overlay)

    def __repr__(self) -> str:
        return f"Polygons(n={len(self.rings)}, crs={self.crs!r})"
