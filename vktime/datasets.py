"""Built-in time-series dataset loaders for vktime."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .series import TimeSeries

_SUNSPOTS_URL = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
_SUNSPOTS_COLS = ["year", "month", "frac_year", "sunspots", "std", "obs", "definitive"]


def load_sunspots(data_dir: str | Path | None = None):
    """Download/cache 277 years of SILSO Monthly Sunspots (1749–present) as ``(df, ts)``."""
    import vkpandas as vpd

    if data_dir is None:
        repo_cache = (
            Path(__file__).resolve().parent.parent / "examples" / "tldr_end2end" / "data"
        )
        cache_dir = repo_cache if repo_cache.exists() else Path("data")
    else:
        cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "sunspots_monthly.csv"
    if not csv_path.exists() or csv_path.stat().st_size < 5000:
        req = urllib.request.Request(
            _SUNSPOTS_URL, headers={"User-Agent": "vkdata-book/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            csv_path.write_bytes(r.read())
    df = vpd.read_csv(csv_path, sep=";", names=_SUNSPOTS_COLS)
    ts = TimeSeries(df["frac_year"], df["sunspots"], name="monthly_sunspots", freq="1M")
    return df, ts


__all__ = ["load_sunspots"]
