"""Terse NYC Taxi example (11.8M trips) using vkpandas + vkshader (pure Vulkan GPU)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT.parent / "vkpandas", _ROOT.parent / "delightful-hertz"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vkpandas as vpd
import vkshader as vks

DEFAULT_PARQUET = Path.home() / ".cache" / "hvsampledata" / "nyc_taxi" / "v2" / "nyc_taxi_wide.parq"


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC Taxi (11.8M trips) with vkpandas + vkshader")
    parser.add_argument("--data", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--how", choices=["eq_hist", "log", "cbrt", "linear"], default="eq_hist")
    parser.add_argument("--out", type=Path, default=Path("/tmp/nyc_taxi_vkshader.jpg"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = vpd.read_parquet(args.data, columns=["pickup_x", "pickup_y"])
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    cvs = vks.Canvas(900, 900, x_range=(-8242000, -8226000), y_range=(4968000, 4985000))
    agg = cvs.points(df, "pickup_x", "pickup_y")
    img = vks.shade(agg, how=args.how, basemap="esri")
    img.save(args.out, quality=90)
    t_render = time.perf_counter() - t1

    print(f"Loaded {len(df):,} trips in {t_load:.3f}s | GPU aggregate + shade + save in {t_render * 1000:.1f} ms -> {args.out}")


if __name__ == "__main__":
    main()
