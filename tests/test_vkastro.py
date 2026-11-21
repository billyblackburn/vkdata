import numpy as np
import vkastro as vka


def test_skycoord_icrs_galactic_and_separation():
    # Galactic Center in ICRS is approximately RA=266.405 deg, Dec=-28.936 deg -> (l≈0, b≈0)
    gc = vka.SkyCoord([266.405], [-28.936], distance=[8178.0], frame="icrs")
    gal = gc.galactic
    l0 = float(gal.ra.to_numpy()[0])
    b0 = float(gal.dec.to_numpy()[0])
    assert (l0 < 0.2 or l0 > 359.8) and abs(b0) < 0.2

    # Round-trip Galactic -> ICRS
    back = gal.icrs
    np.testing.assert_allclose(back.ra.to_numpy(), [266.405], atol=1e-2)
    np.testing.assert_allclose(back.dec.to_numpy(), [-28.936], atol=1e-2)

    # 3D Cartesian norm matches distance
    x, y, z = gc.cartesian()
    r = float(np.sqrt(x.to_numpy()[0]**2 + y.to_numpy()[0]**2 + z.to_numpy()[0]**2))
    assert abs(r - 8178.0) < 1.0

    # Angular separation & cone search
    stars = vka.SkyCoord([0.0, 0.0, 10.0], [0.0, 1.0, 0.0])
    sep = stars.separation(vka.SkyCoord([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])).to_numpy()
    np.testing.assert_allclose(sep, [0.0, 1.0, 10.0], atol=1e-3)
    mask = stars.cone_search(0.0, 0.0, radius_deg=2.0).to_numpy()
    np.testing.assert_array_equal(mask, [True, True, False])


def test_photometry_cosmology_and_fits(tmp_path):
    # Parallax 100 mas -> 10 pc -> M == m
    d = vka.parallax_to_distance([100.0, 10.0], unit="pc").to_numpy()
    np.testing.assert_allclose(d, [10.0, 100.0], rtol=1e-4)
    abs_m = vka.absolute_magnitude([5.0, 5.0], d).to_numpy()
    np.testing.assert_allclose(abs_m, [5.0, 0.0], atol=1e-3)
    lum = vka.luminosity_solar([4.83]).to_numpy()[0]
    assert abs(float(lum) - 1.0) < 1e-3

    # FlatLambdaCDM at z=1.0 (H0=70, Om0=0.3) -> D_L ≈ 6607 Mpc
    cosmo = vka.FlatLambdaCDM(H0=70.0, Om0=0.3)
    dl = float(cosmo.luminosity_distance([1.0]).to_numpy()[0])
    assert 6550.0 < dl < 6660.0

    # Aitoff projection finite bounds
    ax, ay = vka.aitoff_projection([0.0, 180.0, 359.0], [0.0, 45.0, -45.0])
    assert ax.size == 3 and ay.size == 3

    # Minimal valid FITS round-trip
    hdr = (
        f"{'SIMPLE  =                    T':80s}"
        f"{'BITPIX  =                  -32':80s}"
        f"{'NAXIS   =                    2':80s}"
        f"{'NAXIS1  =                    4':80s}"
        f"{'NAXIS2  =                    3':80s}"
        f"{'END':80s}"
    ).ljust(2880).encode("ascii")
    img_np = np.arange(12, dtype=">f4").reshape(3, 4)
    p = tmp_path / "test.fits"
    p.write_bytes(hdr + img_np.tobytes())
    arr_g, meta = vka.read_fits(p)
    assert meta["NAXIS"] == 2
    np.testing.assert_allclose(arr_g.to_numpy(), img_np.astype(np.float32))
