"""Built-in Web Mercator tile sources for vkgeoviews (matching GeoViews tile_sources)."""

from __future__ import annotations

from vkholoviews.elements import Tiles

EsriImagery = Tiles("esri_imagery")
EsriStreet = Tiles("esri_street")
CartoDark = Tiles("carto_dark")
CartoLight = Tiles("carto_light")
OSM = Tiles("osm")

__all__ = [
    "CartoDark",
    "CartoLight",
    "EsriImagery",
    "EsriStreet",
    "OSM",
]
