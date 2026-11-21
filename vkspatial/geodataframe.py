"""GPU GeoDataFrame with 2D Morton Z-order spatial partitioning (SpatialPandas-style)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np
import vknumpy as vnp
import vkpandas as vpd

from .ops import morton_encode, point_in_polygon


class GeoDataFrame:
    """GPU-backed spatial DataFrame with 2D Morton/Z-order partitioning & fast bbox/radius/polygon queries."""

    def __init__(self, df: vpd.DataFrame, x: str = "x", y: str = "y") -> None:
        if not isinstance(df, vpd.DataFrame):
            df = vpd.DataFrame(df)
        if x not in df.columns or y not in df.columns:
            raise KeyError(f"Coordinate columns ({x!r}, {y!r}) not found in DataFrame")
        self.df = df
        self.x_col = x
        self.y_col = y
        # Spatial partition metadata: list of (start, end, xmin, xmax, ymin, ymax)
        self._partitions: Optional[List[Tuple[int, int, float, float, float, float]]] = None

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
