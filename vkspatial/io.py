"""Zero-dependency GeoJSON I/O for vkspatial (Point, Polygon, MultiPolygon FeatureCollections)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import vkpandas as vpd

from .geodataframe import GeoDataFrame
from .ops import lonlat_to_mercator, mercator_to_lonlat


def _project_ring(ring: List[List[float]], to_mercator: bool) -> np.ndarray:
    arr = np.asarray(ring, dtype=np.float32)[:, :2]
    if to_mercator and len(arr) > 0:
        mx, my = lonlat_to_mercator(arr[:, 0], arr[:, 1])
        return np.column_stack([mx.to_numpy(), my.to_numpy()]).astype(np.float32)
    return arr


def read_geojson(
    source: Union[str, Path, Dict[str, Any]],
    to_mercator: bool = False,
    name_prop: str = "name",
) -> Any:
    """Read a GeoJSON file or dict without GDAL/Fiona.

    - For Point FeatureCollections, returns a ``vkspatial.GeoDataFrame``.
    - For Polygon / MultiPolygon FeatureCollections, returns a ``dict[str, np.ndarray]``
      mapping feature name (from ``properties[name_prop]`` or index) to ``(M, 2)`` float32 rings
      ready for ``point_in_polygons()`` or ``gdf.sjoin()``.
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    features = data.get("features", [])
    if not features:
        return {}

    first_type = features[0].get("geometry", {}).get("type", "")
    if first_type == "Point":
        xs, ys = [], []
        prop_cols: Dict[str, List[Any]] = {}
        for feat in features:
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            props = feat.get("properties") or {}
            for k, v in props.items():
                prop_cols.setdefault(k, []).append(v)

        x_arr = np.array(xs, dtype=np.float32)
        y_arr = np.array(ys, dtype=np.float32)
        if to_mercator and len(x_arr) > 0:
            xg, yg = lonlat_to_mercator(x_arr, y_arr)
            cols: Dict[str, Any] = {"x": xg, "y": yg}
            crs = "EPSG:3857"
        else:
            cols = {"x": x_arr, "y": y_arr}
            crs = "EPSG:4326"
        for k, vals in prop_cols.items():
            if len(vals) == len(xs):
                cols[k] = vals
        return GeoDataFrame(vpd.DataFrame(cols), x="x", y="y", crs=crs)

    polygons: Dict[str, np.ndarray] = {}
    for idx, feat in enumerate(features):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        props = feat.get("properties") or {}
        base_name = str(props.get(name_prop, f"poly_{idx}"))
        if gtype == "Polygon":
            rings = geom.get("coordinates", [])
            if rings:
                polygons[base_name] = _project_ring(rings[0], to_mercator=to_mercator)
        elif gtype == "MultiPolygon":
            parts = geom.get("coordinates", [])
            for p_idx, poly_rings in enumerate(parts):
                if poly_rings:
                    key = base_name if len(parts) == 1 else f"{base_name}_{p_idx}"
                    polygons[key] = _project_ring(poly_rings[0], to_mercator=to_mercator)
    return polygons


def to_geojson(
    gdf: GeoDataFrame,
    path: Optional[Union[str, Path]] = None,
    from_mercator: Optional[bool] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Export a point ``GeoDataFrame`` to a GeoJSON FeatureCollection (dict and optional file)."""
    df = gdf.df[:max_rows] if (max_rows is not None and len(gdf.df) > max_rows) else gdf.df
    use_merc = (gdf.crs == "EPSG:3857") if from_mercator is None else bool(from_mercator)

    xs_g = df[gdf.x_col].values.astype(np.float32)
    ys_g = df[gdf.y_col].values.astype(np.float32)
    if use_merc and len(df) > 0:
        lon_g, lat_g = mercator_to_lonlat(xs_g, ys_g)
        lons = lon_g.to_numpy().tolist()
        lats = lat_g.to_numpy().tolist()
    else:
        lons = xs_g.to_numpy().tolist()
        lats = ys_g.to_numpy().tolist()

    prop_names = [c for c in df.columns if c not in (gdf.x_col, gdf.y_col)]
    prop_data = {c: df[c].to_list() for c in prop_names}

    features = []
    for i in range(len(lons)):
        props = {c: prop_data[c][i] for c in prop_names}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lons[i], lats[i]]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": features}
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f)
    return fc
