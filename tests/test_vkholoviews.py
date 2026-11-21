import numpy as np
import vknumpy as vnp
import vkpandas as vpd
import vkholoviews as hv


def test_elements_overlay_layout_and_gpu_histogram():
    rng = np.random.default_rng(42)
    df = vpd.DataFrame({
        "x": rng.uniform(-10.0, 10.0, size=20_000).astype(np.float32),
        "y": rng.uniform(-10.0, 10.0, size=20_000).astype(np.float32),
        "val": rng.uniform(1.0, 50.0, size=20_000).astype(np.float32),
        "hour": rng.integers(0, 12, size=20_000, dtype=np.int32),
    })

    # Overlay (*) with GPU datashade
    pts = hv.datashade(hv.Points(df, ["x", "y"]), how="eq_hist").opts(
        width=200, height=200, xlim=(-10, 10), ylim=(-10, 10), title="GPU Points"
    )
    overlay = hv.Tiles("none") * pts
    assert isinstance(overlay, hv.Overlay)
    assert len(overlay) == 2

    # GPU Histogram + Bars + Curve composed with (+)
    hist = hv.Histogram(df, "val", bins=20, range=(0, 50)).opts(width=240, height=180)
    by_hr = df[["hour", "val"]].groupby("hour").mean()
    bars = hv.Bars(by_hr, "hour", "val").opts(width=240, height=180)
    curve = hv.Curve(by_hr, "hour", "val").opts(width=240, height=180)

    layout = (pts + hist + bars + curve).cols(2)
    assert isinstance(layout, hv.Layout)
    assert len(layout) == 4

    img = layout.render()
    assert img.width > 400 and img.height > 300
    assert img.mode == "RGBA"
