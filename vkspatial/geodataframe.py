"""GPU GeoDataFrame with 2D Morton Z-order spatial partitioning (SpatialPandas-style)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np
import vknumpy as vnp
import vkpandas as vpd

from .ops import (
    lonlat_to_mercator,
    mercator_to_lonlat,
    morton_encode,
    point_in_polygon,
    point_in_polygons,
)


class _CXIndexer:
    """GeoPandas-style spatial coordinate slicer: gdf.cx[xmin:xmax, ymin:ymax]."""

    def __init__(self, gdf: "GeoDataFrame") -> None:
        self._gdf = gdf

    def __getitem__(self, key: Any) -> "GeoDataFrame":
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("cx indexer requires two slices: gdf.cx[xmin:xmax, ymin:ymax]")
        xs, ys = key
        xmin = float(xs.start) if (isinstance(xs, slice) and xs.start is not None) else -1e30
        xmax = float(xs.stop) if (isinstance(xs, slice) and xs.stop is not None) else 1e30
        ymin = float(ys.start) if (isinstance(ys, slice) and ys.start is not None) else -1e30
        ymax = float(ys.stop) if (isinstance(ys, slice) and ys.stop is not None) else 1e30
        sub_df = self._gdf.bbox(xmin, xmax, ymin, ymax)
        return GeoDataFrame(sub_df, x=self._gdf.x_col, y=self._gdf.y_col, crs=self._gdf.crs)


class GeoDataFrame:
    """GPU-backed spatial DataFrame with 2D Morton/Z-order partitioning & GeoPandas-style API."""

    def __init__(self, df: vpd.DataFrame, x: str = "x", y: str = "y", crs: Optional[str] = None) -> None:
        if not isinstance(df, vpd.DataFrame):
            df = vpd.DataFrame(df)
        if x not in df.columns or y not in df.columns:
            raise KeyError(f"Coordinate columns ({x!r}, {y!r}) not found in DataFrame")
        self.df = df
        self.x_col = x
        self.y_col = y
        self.crs = crs
        # Spatial partition metadata: list of (start, end, xmin, xmax, ymin, ymax)
        self._partitions: Optional[List[Tuple[int, int, float, float, float, float]]] = None

    @property
    def cx(self) -> _CXIndexer:
        """GeoPandas-style coordinate-based spatial slicer: ``gdf.cx[xmin:xmax, ymin:ymax]``."""
        return _CXIndexer(self)

    @property
    def total_bounds(self) -> Tuple[float, float, float, float]:
        """Return ``(xmin, ymin, xmax, ymax)`` computed on GPU (GeoPandas convention)."""
        if len(self.df) == 0:
            return (float("nan"), float("nan"), float("nan"), float("nan"))
        xs = self.df[self.x_col].values
        ys = self.df[self.y_col].values
        return (
            float(vnp.nanmin(xs).item()),
            float(vnp.nanmin(ys).item()),
            float(vnp.nanmax(xs).item()),
            float(vnp.nanmax(ys).item()),
        )

    def to_crs(self, crs: str) -> "GeoDataFrame":
        """Reproject coordinates on GPU between ``EPSG:4326`` (lon/lat) and ``EPSG:3857`` (Web Mercator)."""
        target = crs.upper()
        if target == (self.crs or "").upper():
            return GeoDataFrame(self.df.copy(), x=self.x_col, y=self.y_col, crs=target)
        out = self.df.copy()
        if target in ("EPSG:3857", "3857", "MERCATOR"):
            mx, my = lonlat_to_mercator(out[self.x_col], out[self.y_col])
            out[self.x_col] = mx
            out[self.y_col] = my
            return GeoDataFrame(out, x=self.x_col, y=self.y_col, crs="EPSG:3857")
        if target in ("EPSG:4326", "4326", "WGS84", "LONLAT"):
            lon, lat = mercator_to_lonlat(out[self.x_col], out[self.y_col])
            out[self.x_col] = lon
            out[self.y_col] = lat
            return GeoDataFrame(out, x=self.x_col, y=self.y_col, crs="EPSG:4326")
        raise ValueError(f"Unsupported CRS {crs!r}; supported: 'EPSG:4326', 'EPSG:3857'")

    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> List[str]:
        return self.df.columns

    def __getitem__(self, key: Any) -> Any:
        return self.df[key]

    def pack_partitions(self, npartitions: int = 256) -> "GeoDataFrame":
        """Sort rows on GPU by 2D Morton (Z-order) curve and build per-partition bounding boxes."""
        n = len(self.df)
        if n == 0:
            self._partitions = []
            return self

        xg = self.df[self.x_col].values.astype(np.float32)
        yg = self.df[self.y_col].values.astype(np.float32)
        codes = morton_encode(xg, yg)
        order = vnp.argsort(codes)
        self.df = self.df._take_rows(order)

        xs = self.df[self.x_col].values
        ys = self.df[self.y_col].values
        p = max(1, min(int(npartitions), n))
        edges = np.linspace(0, n, p + 1, dtype=np.int64)
        parts: List[Tuple[int, int, float, float, float, float]] = []
        for i in range(p):
            s, e = int(edges[i]), int(edges[i + 1])
            if e <= s:
                continue
            sx, sy = xs[s:e], ys[s:e]
            parts.append((
                s, e,
                float(vnp.min(sx).item()), float(vnp.max(sx).item()),
                float(vnp.min(sy).item()), float(vnp.max(sy).item()),
            ))
        self._partitions = parts
        return self

    def _candidate_ranges(self, xmin: float, xmax: float, ymin: float, ymax: float) -> List[Tuple[int, int]]:
        if not self._partitions:
            return [(0, len(self.df))]
        merged: List[Tuple[int, int]] = []
        for s, e, px0, px1, py0, py1 in self._partitions:
            if px1 < xmin or px0 > xmax or py1 < ymin or py0 > ymax:
                continue
            if merged and merged[-1][1] == s:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged

    def bbox(self, xmin: float, xmax: float, ymin: float, ymax: float) -> vpd.DataFrame:
        """Fast 2D bounding-box query using Morton partition pruning + GPU filtering."""
        ranges = self._candidate_ranges(xmin, xmax, ymin, ymax)
        if not ranges:
            return self.df[:0]
        chunks = []
        for s, e in ranges:
            sub = self.df[s:e]
            m = (
                (sub[self.x_col] >= xmin) & (sub[self.x_col] <= xmax) &
                (sub[self.y_col] >= ymin) & (sub[self.y_col] <= ymax)
            )
            hit = sub[m]
            if len(hit) > 0:
                chunks.append(hit)
        if not chunks:
            return self.df[:0]
        return chunks[0] if len(chunks) == 1 else vpd.concat(chunks)

    def radius_query(self, cx: float, cy: float, radius: float, max_hits: Optional[int] = None) -> vpd.DataFrame:
        """Find points within Euclidean `radius` of (cx, cy) — ideal for click-to-inspect!"""
        cand = self.bbox(cx - radius, cx + radius, cy - radius, cy + radius)
        if len(cand) == 0:
            return cand
        dx = cand[self.x_col] - float(cx)
        dy = cand[self.y_col] - float(cy)
        d2 = dx * dx + dy * dy
        in_r = cand[d2 <= float(radius * radius)]
        if max_hits is not None and len(in_r) > max_hits:
            return in_r[:max_hits]
        return in_r

    def within_polygon(self, polygon: Any) -> vpd.DataFrame:
        """Filter rows whose (x, y) lies inside `polygon` using zero-copy GPU ray-casting."""
        poly_np = np.asarray(polygon, dtype=np.float32)
        xmin, ymin = float(poly_np[:, 0].min()), float(poly_np[:, 1].min())
        xmax, ymax = float(poly_np[:, 0].max()), float(poly_np[:, 1].max())
        cand = self.bbox(xmin, xmax, ymin, ymax)
        if len(cand) == 0:
            return cand
        mask = point_in_polygon(cand[self.x_col], cand[self.y_col], poly_np)
        return cand[mask]

    def sjoin(
        self,
        polygons: Any,
        zone_col: str = "zone_id",
        how: str = "left",
    ) -> "GeoDataFrame":
        """GeoPandas-style single-pass GPU spatial join against K polygons.

        Adds an int32 ``zone_col`` (``0 .. K-1`` or ``-1`` if outside all polygons).
        Set ``how='inner'`` to drop points outside all polygons.
        """
        zones = point_in_polygons(self.df[self.x_col], self.df[self.y_col], polygons)
        out = self.df.copy()
        out[zone_col] = zones
        if how == "inner":
            out = out[out[zone_col] >= 0]
        elif how != "left":
            raise ValueError(f"Unsupported how={how!r}; expected 'left' or 'inner'")
        return GeoDataFrame(out, x=self.x_col, y=self.y_col, crs=self.crs)

    def nearest(self, cx: float, cy: float, k: int = 10) -> vpd.DataFrame:
        """Find the ``k`` nearest points to ``(cx, cy)`` on Vulkan GPU (with distance column)."""
        n = len(self.df)
        if n == 0 or k <= 0:
            return self.df[:0]
        kk = min(int(k), n)
        dx = self.df[self.x_col].values.astype(np.float32) - float(cx)
        dy = self.df[self.y_col].values.astype(np.float32) - float(cy)
        d2 = dx * dx + dy * dy
        # Use GPU top_k on negated d2 for O(N) nearest-k selection
        idx = vnp.top_k(-d2, kk)
        if isinstance(idx, tuple):
            idx = idx[1]
        out = self.df._take_rows(idx)
        out["distance"] = vnp.sqrt(vnp.take(d2, idx))
        return out
