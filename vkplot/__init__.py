"""vkplot: hvPlot-style high-level plotting accessor (.vkplot) for the Vulkan HoloViz stack."""

from __future__ import annotations

from .accessor import vkPlot


def register() -> None:
    """Attach the `.vkplot` property to vkpandas.DataFrame, Series, and vkspatial.GeoDataFrame."""
    try:
        import vkpandas as vpd

        if not hasattr(vpd.DataFrame, "vkplot"):
            vpd.DataFrame.vkplot = property(lambda self: vkPlot(self))
        if not hasattr(vpd.Series, "vkplot"):
            vpd.Series.vkplot = property(lambda self: vkPlot(self))
    except ImportError:
        pass

    try:
        import vkspatial as vksp

        if not hasattr(vksp.GeoDataFrame, "vkplot"):
            vksp.GeoDataFrame.vkplot = property(lambda self: vkPlot(self))
    except ImportError:
        pass


register()

__all__ = ["vkPlot", "register"]
