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


def test_sjoin_crs_cx_nearest_geojson_and_geometry_ops(tmp_path):
    # 1. Single-pass GPU multi-polygon spatial join
    zones = {
        "west": np.array([[-5.0, -5.0], [-1.0, -5.0], [-1.0, 5.0], [-5.0, 5.0]], dtype=np.float32),
        "east": np.array([[ 1.0, -5.0], [ 5.0, -5.0], [ 5.0, 5.0], [ 1.0, 5.0]], dtype=np.float32),
    }
    pts = vpd.DataFrame({
        "x": np.array([-3.0, 0.0, 3.0, 8.0], dtype=np.float32),
        "y": np.array([ 0.0, 0.0, 2.0, 0.0], dtype=np.float32),
    })
    gdf = vksp.GeoDataFrame(pts, x="x", y="y")
    joined = gdf.sjoin(zones, zone_col="zone", how="left")
    np.testing.assert_array_equal(joined["zone"].to_numpy(), np.array([0, -1, 1, -1], dtype=np.int32))
    joined_inner = gdf.sjoin(zones, zone_col="zone", how="inner")
    assert len(joined_inner) == 2

    # 2. cx spatial slicer, total_bounds, and nearest-k
    assert gdf.total_bounds == (-3.0, 0.0, 8.0, 2.0)
    sliced = gdf.cx[-4.0:4.0, -1.0:3.0]
    assert len(sliced) == 3
    nn = gdf.nearest(2.9, 2.0, k=2)
    assert len(nn) == 2
    assert float(nn["distance"].to_numpy().min()) < 0.2

    # 3. CRS conversion (EPSG:4326 <-> EPSG:3857) + GeoJSON round-trip
    geo_df = vksp.GeoDataFrame(
        vpd.DataFrame({"lon": np.array([-73.9855], dtype=np.float32), "lat": np.array([40.7580], dtype=np.float32), "name": ["TS"]}),
        x="lon", y="lat", crs="EPSG:4326",
    )
    merc = geo_df.to_crs("EPSG:3857")
    assert merc.crs == "EPSG:3857"
    back = merc.to_crs("EPSG:4326")
    np.testing.assert_allclose(back["lon"].to_numpy(), [-73.9855], atol=1e-3)

    gj_path = tmp_path / "pts.geojson"
    vksp.to_geojson(merc, gj_path)
    loaded = vksp.read_geojson(gj_path, to_mercator=False)
    assert len(loaded) == 1
    assert loaded.crs == "EPSG:4326"

    # 4. Geometry & trajectory ops (bearing, euclidean, point_to_segment, area, centroid)
    b = vksp.bearing(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32),
                     np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)).to_numpy()[0]
    assert abs(float(b) - 0.0) < 1e-2 or abs(float(b) - 360.0) < 1e-2
    assert abs(float(vksp.euclidean_distance([0.0], [0.0], [3.0], [4.0]).to_numpy()[0]) - 5.0) < 1e-4
    d_seg = vksp.point_to_segment_distance([5.0], [3.0], 0.0, 0.0, 10.0, 0.0).to_numpy()[0]
    assert abs(float(d_seg) - 3.0) < 1e-4
    assert abs(vksp.polygon_area(zones["east"]) - 40.0) < 1e-4
    cx, cy = vksp.polygon_centroid(zones["east"])
    assert abs(cx - 3.0) < 1e-4 and abs(cy - 0.0) < 1e-4


def test_barnes_hut_3d_gpu():
    # 1. Exact symmetric 2-body gravitational attraction
    pm2 = np.array([[-10.0, 0.0, 0.0, 2000.0], [10.0, 0.0, 0.0, 2000.0]], dtype=np.float32)
    vg2 = np.zeros((2, 4), dtype=np.float32)
    sim2 = vksp.BarnesHut3D(pm2, vg2, G=0.085, dt=0.02, theta=0.85, world_half=48.0)
    sim2.step(5)
    out2 = sim2.pos_mass.to_numpy()
    assert out2[0, 0] > -10.0 and out2[1, 0] < 10.0
    np.testing.assert_allclose(out2[0, 0], -out2[1, 0], atol=1e-5)

    # 2. 1,024-body octree + elastic collisions stability & reset
    rng = np.random.default_rng(7)
    n = 1024
    pm = np.zeros((n, 4), dtype=np.float32)
    vg = np.zeros((n, 4), dtype=np.float32)
    pm[:, :3] = rng.uniform(-15.0, 15.0, (n, 3))
    pm[:, 3] = rng.uniform(0.3, 1.0, n)
    pm[0] = [-10.0, 0.0, 0.0, 2000.0]
    pm[1] = [10.0, 0.0, 0.0, 2000.0]

    sim = vksp.BarnesHut3D(pm, vg, G=0.085, dt=0.02, theta=0.85, world_half=48.0)
    sim.step(5)
    assert sim.steps == 5
    out_pm = sim.pos_mass.to_numpy()
    out_vg = sim.vel_gal.to_numpy()
    assert np.isfinite(out_pm).all() and np.isfinite(out_vg).all()
    assert np.linalg.norm(out_pm[:, :3] - pm[:, :3], axis=1).mean() > 1e-3
    # Reset restores initial state cleanly
    sim.reset(pm, vg)
    assert sim.steps == 0
    np.testing.assert_allclose(sim.pos_mass.to_numpy()[0], pm[0], atol=1e-5)
