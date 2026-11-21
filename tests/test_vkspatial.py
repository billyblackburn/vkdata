import numpy as np
import vknumpy as vnp
import vkpandas as vpd
import vkspatial as vksp


def test_projections_and_haversine():
    # NYC Times Square -> JFK Airport (~21.1 km)
    lon = np.array([-73.9855, -73.7781], dtype=np.float32)
    lat = np.array([40.7580, 40.6413], dtype=np.float32)

    x, y = vksp.lonlat_to_mercator(lon, lat)
    lon_rt, lat_rt = vksp.mercator_to_lonlat(x, y)
    np.testing.assert_allclose(lon_rt.to_numpy(), lon, atol=1e-3)
    np.testing.assert_allclose(lat_rt.to_numpy(), lat, atol=1e-3)

    d_km = vksp.haversine_distance(lon[0:1], lat[0:1], lon[1:2], lat[1:2], unit="km").to_numpy()[0]
    assert 20.5 < float(d_km) < 22.5


def test_morton_2d_and_point_in_polygon_gpu():
    rng = np.random.default_rng(11)
    n = 50_000
    x_np = rng.uniform(-10.0, 10.0, size=n).astype(np.float32)
    y_np = rng.uniform(-10.0, 10.0, size=n).astype(np.float32)

    df = vpd.DataFrame({"x": x_np, "y": y_np, "id": np.arange(n, dtype=np.int32)})
    gdf = vksp.GeoDataFrame(df, x="x", y="y").pack_partitions(npartitions=64)

    # Bounding-box query matches exact NumPy ground truth
    box_hit = gdf.bbox(-2.0, 3.0, -1.0, 4.0)
    expected_mask = (x_np >= -2.0) & (x_np <= 3.0) & (y_np >= -1.0) & (y_np <= 4.0)
    assert len(box_hit) == int(expected_mask.sum())

    # Radius query
    rad_hit = gdf.radius_query(0.0, 0.0, radius=2.5)
    expected_rad = (x_np ** 2 + y_np ** 2) <= (2.5 ** 2)
    assert abs(len(rad_hit) - int(expected_rad.sum())) <= 2

    # Point-in-polygon (diamond |x| + |y| <= 4)
    diamond = np.array([[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0]], dtype=np.float32)
    pip_hit = gdf.within_polygon(diamond)
    expected_pip = (np.abs(x_np) + np.abs(y_np)) < 4.0
    assert abs(len(pip_hit) - int(expected_pip.sum())) <= 5
