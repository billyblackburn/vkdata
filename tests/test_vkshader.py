import numpy as np
import vknumpy as vnp
import vkshader as vks


def test_canvas_and_spatial_points_gpu_shading():
    rng = np.random.default_rng(7)
    x = vnp.array(rng.uniform(-10.0, 10.0, size=50_000).astype(np.float32))
    y = vnp.array(rng.uniform(-10.0, 10.0, size=50_000).astype(np.float32))

    cvs = vks.Canvas(128, 128, x_range=(-8.0, 8.0), y_range=(-8.0, 8.0))
    agg_direct = cvs.points({"x": x, "y": y}, "x", "y")

    sp = vks.SpatialPoints(x, y)
    agg_spatial = cvs.points(sp)

    np.testing.assert_array_equal(agg_direct.grid.to_numpy(), agg_spatial.grid.to_numpy())
    assert int(vnp.count_nonzero(agg_direct.grid).item()) > 1000

    for how in ("eq_hist", "log", "cbrt", "linear"):
        img = vks.shade(agg_direct, how=how, basemap=None)
        assert img.rgba.shape == (128, 128, 4)
        assert img.rgba.dtype == np.uint8
        assert img.rgba[:, :, 3].min() == 255
        # Bright pixels should exist where points landed
        assert int(img.rgba[:, :, 0].max()) > 100

    jpg = img.to_jpeg(quality=85)
    assert jpg[:2] == b"\xff\xd8"


def test_canvas_line_raster_and_holoviews_path():
    import vkholoviews as hv

    # Two NaN-separated trajectories: diagonal (cat=0, vel=50) and anti-diagonal (cat=1, vel=250)
    xs = vnp.array(np.array([-5.0, 5.0, np.nan, -5.0, 5.0], dtype=np.float32))
    ys = vnp.array(np.array([-5.0, 5.0, np.nan, 5.0, -5.0], dtype=np.float32))
    vel = vnp.array(np.array([50.0, 50.0, 0.0, 250.0, 250.0], dtype=np.float32))
    cat = vnp.array(np.array([0, 0, 0, 1, 1], dtype=np.int32))
    src = {"x": xs, "y": ys, "vel": vel, "cat": cat}

    cvs = vks.Canvas(64, 64, x_range=(-6.0, 6.0), y_range=(-6.0, 6.0))

    agg_cnt = cvs.line(src, "x", "y", agg="count")
    assert int(vnp.count_nonzero(agg_cnt.grid).item()) > 80
    img_cnt = vks.shade(agg_cnt, how="eq_hist", spread_px=0)
    assert img_cnt.rgba.shape == (64, 64, 4)

    agg_mean = cvs.line(src, "x", "y", agg="mean", column="vel", vmin=0.0, vmax=300.0)
    img_mean = vks.shade(agg_mean, cmap="fire_r")
    assert int(img_mean.rgba[:, :, 0].max()) > 100

    agg_cat = cvs.line(src, "x", "y", agg="by", column="cat", color_key=["red", "blue"])
    img_cat = vks.shade(agg_cat)
    assert int(img_cat.rgba[:, :, 0].max()) > 100  # red channel present
    assert int(img_cat.rgba[:, :, 2].max()) > 100  # blue channel present

    # Declarative vkholoviews.Path + datashade
    pil = hv.datashade(hv.Path(src, ["x", "y"]), agg="by", column="cat", color_key=["red", "blue"]).opts(
        width=64, height=64, xlim=(-6.0, 6.0), ylim=(-6.0, 6.0), title="Paths"
    ).render()
    assert pil.size == (64, 64 + 28)


def test_render_3d_gpu():
    t = np.linspace(0.0, 6.0 * np.pi, 2000, dtype=np.float32)
    src = {
        "x": vnp.array(np.cos(t) * (1.0 + 0.1 * t)),
        "y": vnp.array(np.sin(t) * (1.0 + 0.1 * t)),
        "z": vnp.array(t * 10.0),
        "cat": vnp.array((t > 3.0 * np.pi).astype(np.int32)),
    }
    img = vks.render_3d(
        src, "x", "y", "z",
        width=128, height=96,
        kind="lines", color_by="cat", color_key=["red", "blue"],
        basemap=None,
    )
    assert img.rgba.shape == (96, 128, 4)
    assert img.rgba.dtype == np.uint8
    assert int(img.rgba[:, :, 0].max()) > 120
    assert int(img.rgba[:, :, 2].max()) > 120
