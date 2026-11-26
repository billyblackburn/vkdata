"""Unit tests for vktime (Vulkan GPU TimeSeries, rolling windows, lags, spectral analysis, & anomaly detection)."""

import numpy as np

import vktime


def test_timeseries_operations_and_persistence(tmp_path):
    t = np.arange(0, 100, dtype=np.float32)
    # Linear trend + gentle sine wave
    y = 2.0 * t + 10.0 + 3.0 * np.sin(2.0 * np.pi * t / 10.0)
    ts = vktime.TimeSeries(t, y, name="sensor_temp", freq="1h")
    assert len(ts) == 100 and ts.shape == (100,)

    # Time slicing
    sub = ts.slice_time(20, 49)
    assert len(sub) == 30
    assert sub.timestamps[0] == 20 and sub.timestamps[-1] == 49

    # GPU resampling
    down = ts.resample(step=5, agg="mean")
    assert len(down) == 20
    np.testing.assert_allclose(down.to_numpy()[0], np.mean(y[:5]), atol=1e-4)

    # First difference & linear detrending
    d1 = ts.diff(1)
    assert len(d1) == 99
    det = ts.detrend(method="linear")
    assert abs(float(np.mean(det.to_numpy()))) < 0.2
    assert float(np.std(det.to_numpy())) < 3.0  # trend (std ~58) removed!

    # Normalization
    norm = ts.normalize().to_numpy()
    np.testing.assert_allclose(np.mean(norm), 0.0, atol=1e-4)
    np.testing.assert_allclose(np.std(norm), 1.0, atol=1e-3)

    # Save & load roundtrip
    save_p = tmp_path / "sensor.npz"
    ts.save(save_p)
    ts2 = vktime.TimeSeries.load(save_p)
    assert ts2.name == "sensor_temp" and ts2.freq == "1h"
    np.testing.assert_allclose(ts2.to_numpy(), ts.to_numpy(), atol=1e-5)


def test_rolling_windows_lags_and_temporal_split():
    x = np.arange(1.0, 21.0, dtype=np.float32)
    rm = vktime.rolling_mean(x, window=4).to_numpy()
    assert len(rm) == 20
    np.testing.assert_allclose(rm[-1], np.mean(x[-4:]), atol=1e-5)

    rmin = vktime.rolling_min(x, window=4).to_numpy()
    rmax = vktime.rolling_max(x, window=4).to_numpy()
    np.testing.assert_allclose(rmin[-1], 17.0, atol=1e-5)
    np.testing.assert_allclose(rmax[-1], 20.0, atol=1e-5)

    rsd = vktime.rolling_std(x, window=4).to_numpy()
    np.testing.assert_allclose(rsd[-1], np.std(x[-4:]), atol=1e-4)

    ew = vktime.ewma(x, alpha=0.5).to_numpy()
    assert ew[0] == 1.0 and ew[-1] > ew[0]

    # Supervised lag matrix + rolling feature
    X_gpu, y_gpu = vktime.make_lag_matrix(x, lags=[1, 2, 4], horizon=1, include_rolling=[4])
    X_np, y_np = X_gpu.to_numpy(), y_gpu.to_numpy()
    # start = max_lag - 1 = 3, end = 20 - 1 = 19 -> 16 samples, 3 lags + 1 rolling = 4 cols
    assert X_np.shape == (16, 4) and y_np.shape == (16,)
    # Row 0: anchor index 3 (value 4.0) -> target is index 4 (value 5.0), lag1=4, lag2=3, lag4=1
    np.testing.assert_allclose(X_np[0, :3], [4.0, 3.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(y_np[0], 5.0, atol=1e-5)

    # Sliding windows for 1D CNN (channel_first=True -> (N, 1, W))
    W = vktime.sliding_windows(x, window_size=5, stride=2, channel_first=True).to_numpy()
    assert W.shape == (8, 1, 5)
    np.testing.assert_allclose(W[0, 0], [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-5)

    # Leak-free chronological split
    X_tr, X_te, y_tr, y_te = vktime.temporal_split(X_gpu, y_gpu, train_ratio=0.75)
    assert X_tr.shape[0] == 12 and X_te.shape[0] == 4
    assert float(y_tr.to_numpy()[-1]) < float(y_te.to_numpy()[0])


def test_spectral_periodogram_decomposition_anomalies_and_phase_fold():
    # Pure periodic signal with period = 16 steps + gentle linear trend
    t = np.arange(0, 256, dtype=np.float32)
    period = 16.0
    sig = 10.0 + 0.05 * t + 2.5 * np.sin(2.0 * np.pi * t / period)

    # 1. GPU Autocorrelation should peak strongly at lag = 16
    acf = vktime.autocorrelation(sig - 0.05 * t, max_lag=32).to_numpy()
    assert acf[0] > 0.99
    assert acf[16] > 0.85

    # 2. Periodogram should recover frequency 1/16 = 0.0625
    freqs, power_gpu = vktime.periodogram(sig - 0.05 * t, sample_rate=1.0)
    power = power_gpu.to_numpy()
    peak_freq = float(freqs[int(np.argmax(power[1:])) + 1])
    np.testing.assert_allclose(peak_freq, 1.0 / period, atol=1e-3)

    # 3. Seasonal decomposition recovers trend + seasonal + tiny residual
    trend, seasonal, resid = vktime.seasonal_decompose(sig, period=16)
    assert float(np.std(resid.to_numpy())) < 0.2
    np.testing.assert_allclose(
        trend.to_numpy() + seasonal.to_numpy() + resid.to_numpy(), sig, atol=1e-4
    )

    # 4. Exoplanet-style transit dips: detect_anomalies(direction='low') + phase_fold
    rng = np.random.default_rng(7)
    flux = 1.0 + rng.normal(0.0, 0.001, size=256).astype(np.float32)
    transit_centers = [32, 80, 128, 176, 224]  # every 48 steps!
    for c in transit_centers:
        flux[c - 1 : c + 2] -= 0.025

    _, idx, _ = vktime.detect_anomalies(flux, window=25, threshold=3.5, direction="low")
    for c in transit_centers:
        assert c in idx

    phase, folded_flux = vktime.phase_fold(t, flux, period=48.0, t0=32.0)
    # Near phase 0.0, flux should be noticeably lower than out-of-transit baseline (~1.0)
    in_transit = folded_flux[np.abs(phase) < 0.03]
    out_transit = folded_flux[np.abs(phase) > 0.2]
    assert float(np.mean(in_transit)) < float(np.mean(out_transit)) - 0.015
