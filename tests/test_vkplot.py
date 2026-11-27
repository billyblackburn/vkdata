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


def test_vkplot_reusable_ml_charts_and_vkvis_legend():
    import vkplot
    import vkvis

    x = np.linspace(0, 10, 50, dtype=np.float32)
    im_line = vkplot.line_chart([(x, np.sin(x), "sin"), (x, np.cos(x), "#ff7edb", "cos")], title="Wave", vline=(5.0, "Split"))
    assert im_line.size == (900, 340)

    im_bar = vkplot.barh_chart(["feat_a", "feat_b", "feat_c"], [0.8, -0.4, 0.25], title="Corrs")
    assert im_bar.width == 760 and im_bar.height >= 120

    yt = np.linspace(0, 100, 200, dtype=np.float32)
    yp = yt + np.random.default_rng(0).normal(0, 5, size=200).astype(np.float32)
    im_pa = vkplot.pred_vs_actual(yt, yp, title="Pred vs Actual", lim=(0, 110))
    assert im_pa.size == (396, 380)

    im_cm = vkplot.confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 2], ["Cat", "Dog", "Bird"])
    assert im_cm.width > 200 and im_cm.height > 150

    tiles = [np.random.default_rng(i).random((8, 8)).astype(np.float32) for i in range(6)]
    im_grid = vkplot.image_grid(tiles, titles=[f"T{i}" for i in range(6)], ncols=3, scale=4, cmap="fire", title="Grid")
    assert im_grid.width > 100 and im_grid.height > 80

    im_sbs = vkplot.side_by_side(im_pa, im_pa)
    assert im_sbs.width > im_pa.width * 2

    Y = np.random.default_rng(1).standard_normal((60, 2)).astype(np.float32)
    labs = np.repeat([0, 1, 2], 20)
    im_emb = vkvis.render_embedding(
        Y, labels=labs, class_names=["A", "B", "C"], legend_loc="lower left", clip_percentile=1.5, title="UMAP"
    )
    assert im_emb.width == 640 and im_emb.height == 504

