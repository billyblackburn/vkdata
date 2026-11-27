"""Built-in geospatial dataset loaders for vkspatial."""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

from .geodataframe import GeoDataFrame

_USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv"


def load_earthquakes(data_dir: str | Path | None = None):
    """Download/cache real USGS Global Earthquakes (M >= 2.5, past 30 days) as ``(df, gdf)``.

    Returns a GPU ``vkpandas.DataFrame`` and a ``vkspatial.GeoDataFrame`` with columns
    ``latitude``, ``longitude``, ``depth``, ``mag``, ``rms``, ``place``, and ``mag_type``.
    """
    import vkpandas as vpd

    if data_dir is None:
        repo_cache = Path(__file__).resolve().parent.parent / "examples" / "data"
        cache_dir = repo_cache if repo_cache.exists() else Path("data")
    else:
        cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "usgs_earthquakes_2.5_month.csv"
    if not csv_path.exists() or csv_path.stat().st_size < 5000:
        req = urllib.request.Request(_USGS_URL, headers={"User-Agent": "vkdata-book/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            csv_path.write_bytes(r.read())

    lats, lons, depths, mags, rmss, places, mtypes = [], [], [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                dep = float(row["depth"])
                mag = float(row["mag"])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                rms = float(row.get("rms") or 0.5)
            except ValueError:
                rms = 0.5
            lats.append(lat)
            lons.append(lon)
            depths.append(max(0.0, dep))
            mags.append(mag)
            rmss.append(rms)
            places.append((row.get("place") or "Unknown").strip())
            mtypes.append((row.get("magType") or "ml").strip())

    df = vpd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "depth": depths,
            "mag": mags,
            "rms": rmss,
            "place": places,
            "mag_type": mtypes,
        }
    )
    gdf = GeoDataFrame(df, x="longitude", y="latitude", crs="EPSG:4326")
    return df, gdf


__all__ = ["load_earthquakes"]
