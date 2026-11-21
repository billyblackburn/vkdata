import math
import numpy as np
import vknumpy as vnp
import vkscipy as vsp


def test_special_stats_and_spatial_distance():
    # special: expit, erf, ndtr, softmax, logsumexp
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(vsp.special.expit(x).to_numpy(), 1.0 / (1.0 + np.exp(-x)), atol=1e-5)
    np.testing.assert_allclose(
        vsp.special.erf(x).to_numpy(),
        [math.erf(v) for v in x],
        atol=1e-5,
    )
    np.testing.assert_allclose(vsp.special.ndtr([0.0]).to_numpy(), [0.5], atol=1e-5)
    sm = vsp.special.softmax([[1.0, 2.0, 3.0]]).to_numpy()
    assert abs(float(sm.sum()) - 1.0) < 1e-5

    # stats: norm, zscore, pearsonr, corrcoef
    np.testing.assert_allclose(vsp.stats.norm.cdf([0.0]).to_numpy(), [0.5], atol=1e-5)
    r, p = vsp.stats.pearsonr([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert abs(r - 1.0) < 1e-5 and p < 1e-3
    C = vsp.stats.corrcoef(np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=np.float32)).to_numpy()
    np.testing.assert_allclose(C, np.ones((2, 2)), atol=1e-4)

    # spatial.distance: cdist, pdist, squareform
    A = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    D = vsp.spatial.distance.cdist(A, B, metric="euclidean").to_numpy()
    np.testing.assert_allclose(D, [[3.0, 4.0], [4.0, 3.0]], atol=1e-4)
    pd = vsp.spatial.distance.pdist(A)
    np.testing.assert_allclose(pd.to_numpy(), [5.0], atol=1e-4)
    sq = vsp.spatial.distance.squareform(pd).to_numpy()
    np.testing.assert_allclose(sq, [[0.0, 5.0], [5.0, 0.0]], atol=1e-4)


def test_ndimage_signal_and_optimize():
    # ndimage: gaussian_filter & sobel
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:22, 10:22] = 1.0
    blur = vsp.ndimage.gaussian_filter(img, sigma=1.5).to_numpy()
    assert blur.shape == (32, 32) and 0.4 < blur[16, 16] <= 1.01
    edges = vsp.ndimage.sobel(img).to_numpy()
    assert edges[10, 16] > 1.0 and edges[16, 16] < 1e-4

    # signal: 1D convolve & spectrogram
    sig = np.sin(2.0 * np.pi * 50.0 * np.linspace(0, 1, 2048, dtype=np.float32))
    conv = vsp.signal.convolve(sig, [0.25, 0.5, 0.25], mode="same")
    assert conv.size == 2048
    freqs, times, Sxx = vsp.signal.spectrogram(sig, fs=2048.0, nperseg=256, noverlap=128)
    peak_f = float(freqs.to_numpy()[int(np.argmax(Sxx.to_numpy().mean(axis=1)))])
    assert abs(peak_f - 50.0) < 10.0

    # optimize: curve_fit on GPU (y = a * exp(-b * x) + c)
    xg = vnp.linspace(0.0, 4.0, 200, dtype=np.float32)
    yg = 2.5 * vnp.exp(-1.3 * xg) + 0.5
    popt, _ = vsp.optimize.curve_fit(lambda x, a, b, c: a * vnp.exp(-b * x) + c, xg, yg, p0=[1.0, 0.5, 0.0])
    np.testing.assert_allclose(popt, [2.5, 1.3, 0.5], atol=1e-2)
