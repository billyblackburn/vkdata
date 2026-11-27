"""Tests for Vulkan VRAM profiling (vknumpy) and zero-code-change accelerators (vkpandas.accel, vklearn.accel)."""

from __future__ import annotations

import sys

import vklearn.accel as vkl_accel
import vknumpy as vnp
import vkpandas.accel as vpd_accel


def test_vram_memory_profiling_and_empty_cache():
    vnp.empty_cache()
    vnp.reset_peak_memory_stats()
    base = vnp.memory_allocated()

    with vnp.profile_memory("vram_unit_test", verbose=False) as prof:
        a = vnp.zeros((256, 256))  # 256 * 256 * 4 = 262,144 bytes (0.25 MB)
        b = vnp.ones((256, 256))
        c = a + b
        live_mid = vnp.memory_allocated()

    assert live_mid >= base + 3 * 256 * 256 * 4
    assert prof["delta_bytes"] >= 3 * 256 * 256 * 4
    assert prof["peak_bytes"] >= live_mid
    assert prof["allocs"] >= 3

    info = vnp.memory_info()
    assert info["allocated"] == vnp.memory_allocated()
    assert info["peak"] == vnp.max_memory_allocated()
    assert info["live_buffers"] >= 3
    summary = vnp.memory_summary()
    assert "Vulkan VRAM Summary" in summary
    assert "Live Allocated" in summary

    del a, b, c
    vnp.empty_cache()
    assert vnp.memory_allocated() <= base + 1024


def test_vkpandas_zero_code_change_accel():
    assert not vpd_accel.is_installed()
    vpd_accel.reset_stats()
    vpd_accel.install()
    try:
        assert vpd_accel.is_installed()
        with vpd_accel.profile(verbose=False) as prof:
            import pandas as pd

            df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
            df2 = pd.DataFrame({"x": [4.0, 5.0], "y": [40.0, 50.0]})
            df = pd.concat([df1, df2])
            s = pd.Series([1.0, 2.0, 3.0], name="z")

        assert len(df) == 5
        assert abs(df["y"].mean() - 30.0) < 1e-4
        assert abs(s.mean() - 2.0) < 1e-4
        assert prof["gpu_calls"] >= 4
        assert prof["cpu_fallbacks"] == 0
    finally:
        vpd_accel.uninstall()

    assert not vpd_accel.is_installed()
    assert "pandas" not in sys.modules or not getattr(sys.modules["pandas"], "__vkdata_accel__", False)


def test_vklearn_zero_code_change_accel():
    assert not vkl_accel.is_installed()
    vkl_accel.reset_stats()
    vkl_accel.install()
    try:
        assert vkl_accel.is_installed()
        with vkl_accel.profile(verbose=False) as prof:
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            X = vnp.array([[float(i), float(i * 2)] for i in range(40)])
            y = vnp.array([3.0 * i + 5.0 for i in range(40)])
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
            pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=0.1))])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            score = float(r2_score(y_te, preds))

        assert score > 0.99
        assert prof["gpu_calls"] >= 4
        assert prof["cpu_fallbacks"] == 0
    finally:
        vkl_accel.uninstall()

    assert not vkl_accel.is_installed()
    assert "sklearn" not in sys.modules or not getattr(sys.modules["sklearn"], "__vkdata_accel__", False)
