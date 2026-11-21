"""NYC Taxi Vulkan HoloViz Dashboard (vknumpy + vkpandas + vkshader + vkholoviews)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT.parent / "vkshader", _ROOT.parent / "vkpandas", _ROOT.parent / "delightful-hertz"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vknumpy as vnp
import vkpandas as vpd
import vkholoviews as hv

PARQUET = Path.home() / ".cache" / "hvsampledata" / "nyc_taxi" / "v2" / "nyc_taxi_wide.parq"


def main() -> None:
    t0 = time.perf_counter()
    df = vpd.read_parquet(
        PARQUET,
        columns=["pickup_x", "pickup_y", "fare_amount", "tip_amount", "pickup_hour"],
    )
    df["pickup_hour"] = df["pickup_hour"].astype(vnp.int32)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    clean = df[df["fare_amount"] > 0.0]
    clean["tip_pct"] = (clean["tip_amount"] / clean["fare_amount"]) * 100.0
    by_hour = clean[["pickup_hour", "fare_amount", "tip_pct"]].groupby("pickup_hour").mean()
    t_agg = time.perf_counter() - t1

    t2 = time.perf_counter()
    taxi_map = (
        hv.Tiles("esri")
        * hv.datashade(hv.Points(df, ["pickup_x", "pickup_y"]), how="eq_hist")
    ).opts(
        width=560,
        height=560,
        xlim=(-8242000, -8226000),
        ylim=(4968000, 4985000),
        title="NYC Taxi Pickups (11.84M trips, Vulkan GPU)",
    )

    fare_hist = hv.Histogram(clean, "fare_amount", bins=45, range=(0, 60)).opts(
        width=560, height=280, title="Fare Distribution ($0-$60, GPU bincount)"
    )
    hourly_fare = hv.Bars(by_hour, "pickup_hour", "fare_amount").opts(
        width=560, height=280, title="Mean Fare ($) by Pickup Hour (GPU groupby)"
    )
    hourly_tip = hv.Curve(by_hour, "pickup_hour", "tip_pct").opts(
        width=560, height=280, title="Mean Tip % by Pickup Hour (GPU groupby)"
    )

    dashboard = (taxi_map + fare_hist + hourly_fare + hourly_tip).cols(2)
    out = dashboard.save("/tmp/nyc_taxi_vkholoviews.png")
    t_render = time.perf_counter() - t2

    print(
        f"Loaded {len(df):,} trips in {t_load:.2f}s | "
        f"GPU analytics in {t_agg:.2f}s | "
        f"Rendered 4-panel HoloViz layout in {t_render:.2f}s -> {out}"
    )


if __name__ == "__main__":
    main()
