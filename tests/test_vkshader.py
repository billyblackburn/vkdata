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
