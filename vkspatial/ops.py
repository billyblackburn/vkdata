"""GPU geospatial projections, Haversine distance, Morton Z-order, and point-in-polygon."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import numpy as np
import vknumpy as vnp

from ._backend import morton_2d_gpu, point_in_polygon_gpu, sjoin_polygons_gpu

EARTH_RADIUS_M = 6378137.0
EARTH_RADIUS_KM = 6371.0088


def _to_f32(a: Any) -> vnp.ndarray:
    if hasattr(a, "values") and isinstance(a.values, vnp.ndarray):
        return a.values.astype(np.float32).ravel()
    return vnp.asarray(a, dtype=np.float32).ravel()


def lonlat_to_mercator(lon: Any, lat: Any) -> Tuple[vnp.ndarray, vnp.ndarray]:
    """Project WGS84 (lon, lat) degrees -> EPSG:3857 Web Mercator (x, y) meters on GPU."""
    lon_g = _to_f32(lon)
    lat_g = vnp.clip(_to_f32(lat), -85.05112878, 85.05112878)
    deg2rad = float(math.pi / 180.0)
    x = lon_g * (deg2rad * EARTH_RADIUS_M)
    y = vnp.log(vnp.tan(float(math.pi / 4.0) + lat_g * (deg2rad * 0.5))) * EARTH_RADIUS_M
    return x, y


def mercator_to_lonlat(x: Any, y: Any) -> Tuple[vnp.ndarray, vnp.ndarray]:
    """Project EPSG:3857 Web Mercator (x, y) meters -> WGS84 (lon, lat) degrees on GPU."""
    xg = _to_f32(x)
    yg = _to_f32(y)
    rad2deg = float(180.0 / math.pi)
    lon = (xg / EARTH_RADIUS_M) * rad2deg
    lat = (2.0 * vnp.arctan(vnp.exp(yg / EARTH_RADIUS_M)) - float(math.pi / 2.0)) * rad2deg
    return lon, lat


def haversine_distance(lon1: Any, lat1: Any, lon2: Any, lat2: Any, unit: str = "km") -> vnp.ndarray:
    """Great-circle Haversine distance on Vulkan GPU ('km', 'mi', or 'm')."""
    deg2rad = float(math.pi / 180.0)
    p1 = _to_f32(lat1) * deg2rad
    p2 = _to_f32(lat2) * deg2rad
    dp = p2 - p1
    dl = (_to_f32(lon2) - _to_f32(lon1)) * deg2rad

    s_dp = vnp.sin(dp * 0.5)
    s_dl = vnp.sin(dl * 0.5)
    a = s_dp * s_dp + vnp.cos(p1) * vnp.cos(p2) * (s_dl * s_dl)
    a = vnp.clip(a, 0.0, 1.0)
    c = 2.0 * vnp.arcsin(vnp.sqrt(a))

    if unit == "km":
        return c * EARTH_RADIUS_KM
    if unit == "mi":
        return c * (EARTH_RADIUS_KM * 0.62137119)
    if unit == "m":
        return c * (EARTH_RADIUS_KM * 1000.0)
    raise ValueError(f"Unsupported unit={unit!r}; expected 'km', 'mi', or 'm'")


def morton_encode(
    x: Any,
    y: Any,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> vnp.ndarray:
    """Compute 24-bit 2D Morton (Z-order) spatial codes on GPU."""
    xg = _to_f32(x)
    yg = _to_f32(y)
    if x_range is None:
        xmin, xmax = float(vnp.nanmin(xg).item()), float(vnp.nanmax(xg).item())
        if xmax <= xmin:
            xmax = xmin + 1.0
    else:
        xmin, xmax = float(x_range[0]), float(x_range[1])
    if y_range is None:
        ymin, ymax = float(vnp.nanmin(yg).item()), float(vnp.nanmax(yg).item())
        if ymax <= ymin:
            ymax = ymin + 1.0
    else:
        ymin, ymax = float(y_range[0]), float(y_range[1])
    return morton_2d_gpu(xg, yg, xmin, xmax, ymin, ymax)


def point_in_polygon(x: Any, y: Any, polygon: Any) -> vnp.ndarray:
    """GPU even-odd ray-casting point-in-polygon test; returns bool_ ndarray."""
    xg = _to_f32(x)
    yg = _to_f32(y)
    poly_np = np.asarray(polygon, dtype=np.float32)
    if poly_np.ndim != 2 or poly_np.shape[1] != 2:
        raise ValueError(f"polygon must have shape (M, 2), got {poly_np.shape}")
    poly_g = vnp.array(poly_np.ravel(), dtype=np.float32)
    mask_i32 = point_in_polygon_gpu(xg, yg, poly_g)
    return mask_i32 != 0


def point_in_polygons(x: Any, y: Any, polygons: Any) -> vnp.ndarray:
    """Single-pass GPU spatial join of (x, y) against K polygons.

    Returns an int32 GPU array of matching polygon indices ``0 .. K-1`` (or ``-1`` if outside all).
    """
    xg = _to_f32(x)
    yg = _to_f32(y)
    poly_list = list(polygons.values()) if isinstance(polygons, dict) else list(polygons)
    if not poly_list:
        return vnp.full((xg.size,), -1, dtype=np.int32)

    verts_parts = []
    offsets = [0]
    bboxes = []
    for poly in poly_list:
        p = np.asarray(poly, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
            raise ValueError(f"Each polygon must have shape (M>=3, 2), got {p.shape}")
        verts_parts.append(p)
        offsets.append(offsets[-1] + len(p))
        bboxes.extend([float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())])

    verts_g = vnp.array(np.vstack(verts_parts).ravel(), dtype=np.float32)
    offs_g = vnp.array(np.array(offsets, dtype=np.int32), dtype=np.int32)
    bbox_g = vnp.array(np.array(bboxes, dtype=np.float32), dtype=np.float32)
    return sjoin_polygons_gpu(xg, yg, verts_g, offs_g, bbox_g, len(poly_list))


def euclidean_distance(x1: Any, y1: Any, x2: Any, y2: Any) -> vnp.ndarray:
    """Planar 2D Euclidean distance on Vulkan GPU."""
    dx = _to_f32(x2) - _to_f32(x1)
    dy = _to_f32(y2) - _to_f32(y1)
    return vnp.sqrt(dx * dx + dy * dy)


def bearing(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> vnp.ndarray:
    """Initial compass bearing in degrees [0, 360) from (lon1, lat1) to (lon2, lat2) on GPU."""
    deg2rad = float(math.pi / 180.0)
    rad2deg = float(180.0 / math.pi)
    p1 = _to_f32(lat1) * deg2rad
    p2 = _to_f32(lat2) * deg2rad
    dl = (_to_f32(lon2) - _to_f32(lon1)) * deg2rad
    y = vnp.sin(dl) * vnp.cos(p2)
    x = vnp.cos(p1) * vnp.sin(p2) - vnp.sin(p1) * vnp.cos(p2) * vnp.cos(dl)
    # atan2 via numpy fallback only if atan2 unavailable; check vnp.arctan2
    if hasattr(vnp, "arctan2"):
        ang = vnp.arctan2(y, x) * rad2deg
    else:
        ang = vnp.array(np.degrees(np.arctan2(y.to_numpy(), x.to_numpy())).astype(np.float32), dtype=np.float32)
    return (ang + 360.0) % 360.0


def point_to_segment_distance(px: Any, py: Any, ax: float, ay: float, bx: float, by: float) -> vnp.ndarray:
    """Shortest planar distance on GPU from points (px, py) to line segment (ax, ay)-(bx, by)."""
    pxg = _to_f32(px)
    pyg = _to_f32(py)
    abx = float(bx - ax)
    aby = float(by - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        dx = pxg - float(ax)
        dy = pyg - float(ay)
        return vnp.sqrt(dx * dx + dy * dy)
    t = ((pxg - float(ax)) * abx + (pyg - float(ay)) * aby) * (1.0 / ab2)
    t = vnp.clip(t, 0.0, 1.0)
    proj_x = float(ax) + t * abx
    proj_y = float(ay) + t * aby
    dx = pxg - proj_x
    dy = pyg - proj_y
    return vnp.sqrt(dx * dx + dy * dy)


def polygon_area(polygon: Any) -> float:
    """Unsigned planar polygon area (shoelace formula) in coordinate units squared."""
    p = np.asarray(polygon, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(polygon: Any) -> Tuple[float, float]:
    """Planar polygon centroid (cx, cy)."""
    p = np.asarray(polygon, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        return (0.0, 0.0)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a2 = float(cross.sum())
    if abs(a2) < 1e-12:
        return (float(x.mean()), float(y.mean()))
    cx = float(((x + x1) * cross).sum() / (3.0 * a2))
    cy = float(((y + y1) * cross).sum() / (3.0 * a2))
    return (cx, cy)
