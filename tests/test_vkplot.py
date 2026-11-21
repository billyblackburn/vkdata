import numpy as np
import vkpandas as vpd
import vkspatial as vksp
import vkplot.vkpandas  # noqa: F401 - registers .vkplot accessor


def test_vkplot_dataframe_series_and_geodataframe():
    rng = np.random.default_rng(123)
    n = 4000
    df = vpd.DataFrame({
        "x": rng.uniform(-5.0, 5.0, size=n).astype(np.float32),
        "y": rng.uniform(-5.0, 5.0, size=n).astype(np.float32),
        "z": rng.uniform(0.0, 100.0, size=n).astype(np.float32),
        "cat": rng.integers(0, 2, size=n, dtype=np.int32),
    })

    # Points / scatter
    p = df.vkplot.points("x", "y", width=128, height=128, title="Pts")
    assert p.render().size == (128, 128 + 28)
    s = df.vkplot.scatter("x", "y", width=96, height=96)
    assert s.render().size == (96, 96)

    # Paths (count, mean, categorical by=)
    pc = df.vkplot.paths("x", "y", width=96, height=96)
    assert pc.render().size == (96, 96)
    pm = df.vkplot.paths("x", "y", c="z", cmap="fire_r", width=96, height=96)
    assert pm.render().size == (96, 96)
    pb = df.vkplot.paths("x", "y", by="cat", color_key=["red", "blue"], width=96, height=96)
    assert pb.render().size == (96, 96)

    # Histogram on DataFrame and Series
    h1 = df.vkplot.hist("z", bins=20, range=(0, 100), width=240, height=160)
    assert h1.render().size == (240, 160)
    h2 = df["z"].vkplot.hist(bins=20, range=(0, 100), width=240, height=160)
    assert h2.render().size == (240, 160)

    # Bar & Line on aggregated DataFrame
    agg_df = vpd.DataFrame({"hour": np.arange(6, dtype=np.int32), "val": np.linspace(10.0, 30.0, 6, dtype=np.float32)})
    assert agg_df.vkplot.bar("hour", "val", width=200, height=140).render().size == (200, 140)
    assert agg_df.vkplot.line("hour", "val", width=200, height=140).render().size == (200, 140)

    # Composable Layout via +
    layout = (p + h1).cols(2)
    assert layout.render().width > 300

    # GeoDataFrame default (x, y)
    gdf = vksp.GeoDataFrame(df, x="x", y="y")
    assert gdf.vkplot.points(width=96, height=96).render().size == (96, 96)

    # 3D Vulkan GPU plot
    img3d = df.vkplot.plot3d("x", "y", "z", width=128, height=96, basemap=None)
    assert img3d.rgba.shape == (96, 128, 4)
