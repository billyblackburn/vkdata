import numpy as np
import vknumpy as vnp
import vkshader as vks
import vkcolorcet as cc


def test_vkcolorcet_palettes_and_gpu_shading():
    assert len(cc.fire) == 256
    assert len(cc.viridis) == 256
    assert len(cc.glasbey) >= 16

    lut = cc.get_lut_u32("viridis")
    assert lut.shape == (256,) and lut.dtype == np.uint32

    # Shade same GPU grid with fire vs kbc (blue/cyan) and verify distinct RGB channels
    rng = np.random.default_rng(5)
    x = vnp.array(rng.normal(0.0, 1.0, size=20_000).astype(np.float32))
    y = vnp.array(rng.normal(0.0, 1.0, size=20_000).astype(np.float32))
    cvs = vks.Canvas(96, 96, x_range=(-3.0, 3.0), y_range=(-3.0, 3.0))
    agg = cvs.points({"x": x, "y": y}, "x", "y")

    img_fire = vks.shade(agg, cmap="fire", basemap=None)
    img_kbc = vks.shade(agg, cmap=cc.kbc, basemap=None)
    assert img_fire.rgba.shape == (96, 96, 4)
    # kbc should have much higher mean blue than red compared to fire
    assert img_kbc.rgba[:, :, 2].mean() > img_kbc.rgba[:, :, 0].mean()
    assert img_fire.rgba[:, :, 0].mean() > img_fire.rgba[:, :, 2].mean()
