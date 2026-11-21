"""vkspatial: SpatialPandas / cuSpatial-style Vulkan GPU spatial indexing & queries."""

from .geodataframe import GeoDataFrame
from .ops import (
    haversine_distance,
    lonlat_to_mercator,
    mercator_to_lonlat,
    morton_encode,
    point_in_polygon,
)

__all__ = [
    "GeoDataFrame",
    "haversine_distance",
    "lonlat_to_mercator",
    "mercator_to_lonlat",
    "morton_encode",
    "point_in_polygon",
]
