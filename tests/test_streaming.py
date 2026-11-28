"""Tests for Polars/Dask-cuDF style out-of-core GPU streaming in vkpandas."""

from __future__ import annotations

import numpy as np

import vkpandas as vpd


def test_read_csv_chunksize_and_scan_csv_streaming(tmp_path):
    csv_path = tmp_path / "big_stream.csv"
    rng = np.random.default_rng(42)
    n = 12_000
    regions = np.array(["north", "south", "east", "west"])[rng.integers(0, 4, size=n)]
    prices = rng.uniform(10.0, 500.0, size=n).astype(np.float32)
    qty = rng.integers(1, 10, size=n).astype(np.int32)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("region,price,qty\n")
        for r, p, q in zip(regions, prices, qty):
            f.write(f"{r},{p:.4f},{q}\n")

    # 1. pandas-style read_csv(..., chunksize=2500)
    chunks = list(vpd.read_csv(csv_path, chunksize=2500))
    assert len(chunks) == 5
    assert sum(len(c) for c in chunks) == n

    # 2. Polars-style scan_csv() exact Welford agg() & describe()
    sf = vpd.scan_csv(csv_path, chunksize=3000)
    res = sf.agg({"price": ["count", "sum", "mean", "std", "min", "max"]})
    assert int(res["price_count"]) == n
    assert abs(res["price_mean"] - float(np.mean(prices))) < 1e-2
    assert abs(res["price_std"] - float(np.std(prices, ddof=1))) < 1e-2
    assert abs(res["price_min"] - float(np.min(prices))) < 1e-2
    assert abs(res["price_max"] - float(np.max(prices))) < 1e-2
    assert sf.last_stats["chunks"] == 4
    assert sf.last_stats["rows_in"] == n

    # 3. Lazy filter + with_columns + streaming groupby().agg()
    def add_rev(df):
        df["revenue"] = df["price"] * df["qty"]
        return df

    gb_df = (
        vpd.scan_csv(csv_path, chunksize=4000)
        .filter(lambda df: df["price"] >= 100.0)
        .with_columns(add_rev)
        .groupby("region")
        .agg({"revenue": ["sum", "mean", "count"]})
    )
    mask = prices >= 100.0
    assert len(gb_df) == 4
    for r_name, rev_sum, rev_mean, cnt in zip(
        gb_df["region"].to_list(),
        gb_df["revenue_sum"].to_list(),
        gb_df["revenue_mean"].to_list(),
        gb_df["revenue_count"].to_list(),
    ):
        rmask = mask & (regions == r_name)
        expected_rev = prices[rmask] * qty[rmask]
        assert int(cnt) == int(np.sum(rmask))
        assert abs(rev_sum - float(np.sum(expected_rev))) / max(1.0, float(np.sum(expected_rev))) < 1e-3
        assert abs(rev_mean - float(np.mean(expected_rev))) < 0.25

    # 4. Streaming to_parquet() + auto VRAM budget
    assert vpd.recommended_vram_budget_mb() >= 32.0
    pq_out = tmp_path / "filtered.parquet"
    vpd.scan_csv(csv_path, vram_budget_mb="auto").filter(lambda df: df["qty"] >= 5).to_parquet(pq_out)
    loaded = vpd.read_parquet(pq_out)
    assert len(loaded) == int(np.sum(qty >= 5))
