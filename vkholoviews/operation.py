"""GPU datashade() / rasterize() operations wrapping vkshader."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import vkshader as vks

from .core import Element
from .elements import Points, _col_to_vk


class DatashadedPoints(Element):
    """A Points Element wrapped with a lazy GPU vkshader SpatialPoints index."""

    def __init__(
        self,
        points: Points,
        how: str = "eq_hist",
        alpha: int = 235,
        **opts: Any,
    ) -> None:
        merged_opts = {**points._opts, **opts}
        super().__init__(data=points.data, kdims=points.kdims, vdims=points.vdims, opts=merged_opts)
        self.points = points
        self.how = how
        self.alpha = int(alpha)
        self._spatial: Optional[vks.SpatialPoints] = None

    def get_spatial(self) -> vks.SpatialPoints:
        if self._spatial is None:
            xs = _col_to_vk(self.data, self.kdims[0])
            ys = _col_to_vk(self.data, self.kdims[1])
            self._spatial = vks.SpatialPoints(xs, ys)
        return self._spatial

    def render_gpu(
        self,
        width: int,
        height: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        basemap: Optional[str] = None,
        how: Optional[str] = None,
    ) -> vks.ShadedImage:
        cvs = vks.Canvas(width, height, x_range=x_range, y_range=y_range)
        sp = self.get_spatial()
        agg = cvs.points(sp, x_range=x_range, y_range=y_range)
        return vks.shade(agg, how=how or self.how, alpha=self.alpha, basemap=basemap)

    def __repr__(self) -> str:
        return f"DatashadedPoints({self.kdims}, how={self.how!r})"


def datashade(
    element: Points,
    how: str = "eq_hist",
    alpha: int = 235,
    **opts: Any,
) -> DatashadedPoints:
    """Wrap a Points element for zero-copy Vulkan GPU datashading via vkshader."""
    if not isinstance(element, Points):
        raise TypeError(f"datashade() expects a Points element, got {type(element).__name__}")
    return DatashadedPoints(element, how=how, alpha=alpha, **opts)


def rasterize(
    element: Points,
    how: str = "eq_hist",
    alpha: int = 235,
    **opts: Any,
) -> DatashadedPoints:
    """Alias for datashade() matching HoloViews API."""
    return datashade(element, how=how, alpha=alpha, **opts)
