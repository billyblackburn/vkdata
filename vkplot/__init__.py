"""vkplot: hvPlot-style high-level plotting accessor (.vkplot) & reusable SynthWave '84 ML charts."""

from __future__ import annotations

from .accessor import vkPlot
from .charts import (
    barh_chart,
    confusion_matrix,
    image_grid,
    line_chart,
    pred_vs_actual,
    side_by_side,
)


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

__all__ = [
    "barh_chart",
    "confusion_matrix",
    "image_grid",
    "line_chart",
    "pred_vs_actual",
    "register",
    "side_by_side",
    "vkPlot",
]
