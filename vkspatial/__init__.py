"""vkspatial: GeoPandas / SpatialPandas / cuSpatial-style Vulkan GPU spatial indexing & queries."""

from .geodataframe import GeoDataFrame
from .io import read_geojson, to_geojson

read_file = read_geojson  # GeoPandas-compatible alias
from .ops import (
    bearing,
    euclidean_distance,
    haversine_distance,
    lonlat_to_mercator,
    mercator_to_lonlat,
    morton_encode,
    point_in_polygon,
    point_in_polygons,
    point_to_segment_distance,
    polygon_area,
    polygon_centroid,
)

__all__ = [
    "GeoDataFrame",
    "bearing",
    "euclidean_distance",
    "haversine_distance",
    "lonlat_to_mercator",
    "mercator_to_lonlat",
    "morton_encode",
    "point_in_polygon",
    "point_in_polygons",
    "point_to_segment_distance",
    "polygon_area",
    "polygon_centroid",
    "read_geojson",
    "to_geojson",
]
