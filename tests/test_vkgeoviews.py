import numpy as np
import vkpandas as vpd
import vkgeoviews as gv


def test_vkgeoviews_crs_points_and_choropleth_polygons():
    # Raw WGS84 (lon, lat) points auto-projected to Web Mercator on GPU
    df = vpd.DataFrame({
        "lon": np.array([-73.99, -73.98, -73.97, -73.96], dtype=np.float32),
        "lat": np.array([ 40.75,  40.76,  40.77,  40.78], dtype=np.float32),
    })
    pts = gv.Points(df, ["lon", "lat"], crs="EPSG:4326")
    # Projected coordinates should be in Web Mercator meters (~ -8.23e6, 4.97e6)
    xs = pts.data["lon"].to_numpy()
    ys = pts.data["lat"].to_numpy()
    assert xs.min() < -8.0e6 and ys.min() > 4.9e6

    zones = {
        "Midtown": [(-74.002, 40.748), (-73.967, 40.748), (-73.961, 40.768), (-73.993, 40.768)],
        "UES":     [(-73.973, 40.764), (-73.944, 40.764), (-73.941, 40.785), (-73.959, 40.785)],
    }
    choropleth = gv.Polygons(zones, values={"Midtown": 14.5, "UES": 12.1}, crs="EPSG:4326").opts(
        width=240, height=200, title="NYC Choropleth"
    )
    img = choropleth.render()
    assert img.size == (240, 200 + 28)

    # Overlay datashaded GPU points + Polygons (inherits title from choropleth)
    combo = (gv.datashade(pts) * choropleth).opts(width=240, height=200)
    assert combo.render().size == (240, 200 + 28)
    assert gv.tile_sources.CartoDark.source == "carto_dark"
    assert gv.tile_sources.EsriImagery.source == "esri_imagery"
