"""hvPlot-style submodule (`import vkplot.vkpandas`) to register the `.vkplot` accessor."""

from . import register, vkPlot

register()

__all__ = ["vkPlot", "register"]
