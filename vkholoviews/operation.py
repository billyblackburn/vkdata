"""GPU datashade() / rasterize() operations wrapping vkshader."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import vkshader as vks

from .core import Element
from .elements import Path, Points, _col_to_vk


class DatashadedPoints(Element):
    """A Points Element wrapped with a lazy GPU vkshader SpatialPoints index."""

    def __init__(
        self,
        points: Points,
        how: str = "eq_hist",
        alpha: int = 235,
        cmap: Any = "fire",
        **opts: Any,
    ) -> None:
        merged_opts = {**points._opts, **opts}
        super().__init__(data=points.data, kdims=points.kdims, vdims=points.vdims, opts=merged_opts)
        self.points = points
        self.how = how
        self.alpha = int(alpha)
        self.cmap = cmap
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
        cmap: Optional[Any] = None,
    ) -> vks.ShadedImage:
        cvs = vks.Canvas(width, height, x_range=x_range, y_range=y_range)
        sp = self.get_spatial()
        agg = cvs.points(sp, x_range=x_range, y_range=y_range)
        return vks.shade(agg, how=how or self.how, alpha=self.alpha, basemap=basemap, cmap=cmap if cmap is not None else self.cmap)

    def __repr__(self) -> str:
        return f"DatashadedPoints({self.kdims}, how={self.how!r})"


class DatashadedPaths(Element):
    """A Path Element rasterized on Vulkan GPU via vkshader.Canvas.line()."""

    def __init__(
        self,
        path: Path,
        how: str = "eq_hist",
        alpha: int = 235,
        agg: str = "count",
        column: Optional[str] = None,
        color_key: Any = None,
        cmap: str = "fire",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        spread_px: int = 0,
        **opts: Any,
    ) -> None:
        merged_opts = {**path._opts, **opts}
        super().__init__(data=path.data, kdims=path.kdims, vdims=path.vdims, opts=merged_opts)
        self.path = path
        self.how = how
        self.alpha = int(alpha)
        self.agg = agg
        self.column = column
        self.color_key = color_key
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.spread_px = int(spread_px)

    def render_gpu(
        self,
        width: int,
        height: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        basemap: Optional[str] = None,
        how: Optional[str] = None,
        spread_px: Optional[int] = None,
        cmap: Optional[Any] = None,
    ) -> vks.ShadedImage:
        cvs = vks.Canvas(width, height, x_range=x_range, y_range=y_range)
        agg_res = cvs.line(
            self.data,
            self.kdims[0],
            self.kdims[1],
            agg=self.agg,
            column=self.column,
            color_key=self.color_key,
            x_range=x_range,
            y_range=y_range,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        sp = self.spread_px if spread_px is None else spread_px
        return vks.shade(
            agg_res,
            how=how or self.how,
            alpha=self.alpha,
            basemap=basemap,
            spread_px=sp,
            cmap=cmap if cmap is not None else self.cmap,
        )

    def __repr__(self) -> str:
        return f"DatashadedPaths({self.kdims}, agg={self.agg!r}, column={self.column!r})"


def datashade(
    element: Any,
    how: str = "eq_hist",
    alpha: int = 235,
    **opts: Any,
) -> Any:
    """Wrap a Points or Path element for zero-copy Vulkan GPU datashading via vkshader."""
    if isinstance(element, Points):
        return DatashadedPoints(element, how=how, alpha=alpha, **opts)
    if isinstance(element, Path):
        return DatashadedPaths(element, how=how, alpha=alpha, **opts)
    raise TypeError(f"datashade() expects a Points or Path element, got {type(element).__name__}")


def rasterize(
    element: Any,
    how: str = "eq_hist",
    alpha: int = 235,
    **opts: Any,
) -> Any:
    """Alias for datashade() matching HoloViews API."""
    return datashade(element, how=how, alpha=alpha, **opts)
