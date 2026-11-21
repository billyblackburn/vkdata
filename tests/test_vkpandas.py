"""End-to-end pytest suite for vkpandas on top of vknumpy (AMD RADV Vulkan GPU)."""

from pathlib import Path
import numpy as np
import vkpandas as vpd


def test_series_ops_and_missing_values():
    s = vpd.Series([10.0, np.nan, 30.0, 40.0, np.nan], name="val")
    assert len(s) == 5
    assert s.count() == 3
    np.testing.assert_allclose(s.sum(), 80.0, rtol=1e-5)
    np.testing.assert_allclose(s.mean(), 80.0 / 3.0, rtol=1e-5)
    np.testing.assert_allclose(s.min(), 10.0)
    np.testing.assert_allclose(s.max(), 40.0)

    np.testing.assert_allclose(s.fillna(0.0).to_numpy(), [10.0, 0.0, 30.0, 40.0, 0.0])
    np.testing.assert_allclose(s.dropna().to_numpy(), [10.0, 30.0, 40.0])
    np.testing.assert_allclose(s.ffill().to_numpy(), [10.0, 10.0, 30.0, 40.0, 40.0])

    # Rolling sum & mean
    r = vpd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    rs = r.rolling(3).sum().to_numpy()
    rm = r.rolling(3).mean().to_numpy()
    assert np.isnan(rs[0]) and np.isnan(rs[1])
    np.testing.assert_allclose(rs[2:], [6.0, 9.0, 12.0])
    np.testing.assert_allclose(rm[2:], [2.0, 3.0, 4.0])


def test_dataframe_filter_sort_and_computed_columns():
    df = vpd.DataFrame({
        "city": ["nyc", "la", "nyc", "chi", "la", "nyc"],
        "price": [10.0, 25.0, 30.0, 15.0, 45.0, 50.0],
        "qty": [2, 1, 3, 4, 2, 1],
    })
    assert df.shape == (6, 3)

    df["total"] = df["price"] * df["qty"]
    big = df[df["total"] >= 50.0].sort_values("total", ascending=False)

    np.testing.assert_allclose(big["total"].to_numpy(), [90.0, 90.0, 60.0, 50.0])
    assert set(big["city"].to_list()) == {"nyc", "la", "chi"}

    # String equality filter & isin
    nyc = df[df["city"] == "nyc"]
    assert len(nyc) == 3
    np.testing.assert_allclose(nyc["price"].to_numpy(), [10.0, 30.0, 50.0])
    assert df["city"].value_counts() == {"nyc": 3, "la": 2, "chi": 1}
    assert len(df[df["city"].isin(["la", "chi"])]) == 3


def test_dataframe_groupby_string_and_int_keys():
    df = vpd.DataFrame({
        "city": ["nyc", "la", "nyc", "chi", "la", "nyc"],
        "sales": [10.0, 20.0, 30.0, 5.0, 15.0, 50.0],
        "units": [1.0, 2.0, 3.0, 1.0, 3.0, 5.0],
    })

    g = df.groupby("city").agg({"sales": "sum", "units": "mean"})
    rows = {c: (s, u) for c, s, u in zip(g["city"].to_list(), g["sales"].to_list(), g["units"].to_list())}
    np.testing.assert_allclose(rows["nyc"], (90.0, 3.0), rtol=1e-4)
    np.testing.assert_allclose(rows["la"], (35.0, 2.5), rtol=1e-4)
    np.testing.assert_allclose(rows["chi"], (5.0, 1.0), rtol=1e-4)


def test_dataframe_merge_inner_join():
    orders = vpd.DataFrame({
        "user_id": [103, 101, 999, 102, 101],
        "amount": [15.0, 20.0, 5.0, 30.0, 10.0],
    })
    users = vpd.DataFrame({
        "user_id": [102, 101, 103],
        "tier": ["gold", "silver", "bronze"],
    })

    joined = orders.merge(users, on="user_id", how="inner")
    assert len(joined) == 4
    np.testing.assert_array_equal(joined["user_id"].to_numpy(), [103, 101, 102, 101])
    np.testing.assert_allclose(joined["amount"].to_numpy(), [15.0, 20.0, 30.0, 10.0])
    assert joined["tier"].to_list() == ["bronze", "silver", "gold", "silver"]


def test_concat_and_read_csv(tmp_path: Path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("city,price,qty\nnyc,10.5,2\nla,20.0,3\nnyc,,4\n", encoding="utf-8")

    df = vpd.read_csv(csv_file)
    assert df.shape == (3, 3)
    assert df["city"].to_list() == ["nyc", "la", "nyc"]
    assert df["price"].count() == 2
    assert len(df.dropna()) == 2

    df2 = vpd.concat([df.dropna(), df.dropna()])
    assert df2.shape == (4, 3)
    np.testing.assert_allclose(df2["price"].to_numpy(), [10.5, 20.0, 10.5, 20.0])
