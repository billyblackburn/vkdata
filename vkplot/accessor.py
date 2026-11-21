"""hvPlot-style .vkplot accessor for vkpandas DataFrame/Series and vkspatial GeoDataFrame."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import vkholoviews as hv
import vkshader as vks


def _normalize_tiles(tiles: Any) -> Optional[str]:
    if not tiles:
        return None
    if tiles is True:
        return "esri"
    s = str(tiles).lower()
    if "esri" in s:
        return "esri"
    return s


def _apply_opts(elem: Any, opts: dict) -> Any:
    if not opts:
        return elem
    return elem.opts(**opts)


class vkPlot:
    """High-level hvPlot-style plotting accessor backed by Vulkan GPU vkholoviews & vkshader."""

    def __init__(self, data: Any) -> None:
        # Unwrap vkspatial.GeoDataFrame cleanly while remembering default (x, y)
        if hasattr(data, "df") and hasattr(data, "x_col") and hasattr(data, "y_col"):
            self._df = data.df
            self._default_x: Optional[str] = data.x_col
            self._default_y: Optional[str] = data.y_col
        else:
            self._df = data
            self._default_x = None
            self._default_y = None

    def _resolve_xy(self, x: Optional[str], y: Optional[str]) -> Tuple[str, str]:
        cols = list(getattr(self._df, "columns", []))
        rx = x or self._default_x or (cols[0] if len(cols) >= 1 else "x")
        ry = y or self._default_y or (cols[1] if len(cols) >= 2 else "y")
        return rx, ry

    def __call__(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        kind: str = "line",
        **kwargs: Any,
    ) -> Any:
        k = kind.lower()
        if k in ("point", "points", "scatter"):
            return self.points(x=x, y=y, **kwargs)
        if k in ("path", "paths"):
            return self.paths(x=x, y=y, **kwargs)
        if k in ("hist", "histogram"):
            return self.hist(y=y or x, **kwargs)
        if k in ("bar", "bars"):
            return self.bar(x=x, y=y, **kwargs)
        if k in ("line", "curve"):
            return self.line(x=x, y=y, **kwargs)
        if k in ("3d", "plot3d"):
            return self.plot3d(x=x, y=y, **kwargs)
        raise ValueError(f"Unsupported kind={kind!r}; expected points, scatter, paths, hist, bar, line, or 3d")

    def points(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        tiles: Any = None,
        datashade: bool = True,
        how: str = "eq_hist",
        alpha: int = 235,
        **opts: Any,
    ) -> Any:
        """2D GPU point cloud / scatter plot (with optional Esri tiles & datashading)."""
        rx, ry = self._resolve_xy(x, y)
        elem: Any = hv.Points(self._df, [rx, ry])
        if datashade:
            elem = hv.datashade(elem, how=how, alpha=alpha)
        t = _normalize_tiles(tiles)
        if t:
            elem = hv.Tiles(t) * elem
        return _apply_opts(elem, opts)

    def scatter(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Alias for points() matching pandas / hvPlot API."""
        return self.points(x=x, y=y, **kwargs)

    def paths(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        tiles: Any = None,
        datashade: bool = True,
        aggregator: str = "count",
        by: Optional[str] = None,
        c: Optional[str] = None,
        color_key: Any = None,
        cmap: str = "fire",
        how: str = "eq_hist",
        alpha: int = 235,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        spread_px: int = 0,
        **opts: Any,
    ) -> Any:
        """2D GPU polyline / trajectory plot (e.g. OpenSky flights, AIS ship tracks)."""
        rx, ry = self._resolve_xy(x, y)
        agg = aggregator
        col = c
        if by is not None:
            agg = "by"
            col = by
        elif c is not None and agg == "count":
            agg = "mean"

        elem: Any = hv.Path(self._df, [rx, ry])
        if datashade:
            elem = hv.datashade(
                elem,
                how=how,
                alpha=alpha,
                agg=agg,
                column=col,
                color_key=color_key,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                spread_px=spread_px,
            )
        t = _normalize_tiles(tiles)
        if t:
            elem = hv.Tiles(t) * elem
        return _apply_opts(elem, opts)

    def hist(
        self,
        y: Optional[str] = None,
        bins: int = 40,
        range: Optional[Tuple[float, float]] = None,
        **opts: Any,
    ) -> hv.Histogram:
        """1D GPU-accelerated Histogram computed via vknumpy.histogram."""
        if not hasattr(self._df, "columns"):
            # vkpandas.Series
            name = getattr(self._df, "name", None) or "value"
            elem = hv.Histogram({name: self._df}, name, bins=bins, range=range)
            return _apply_opts(elem, opts)
        col = y or list(self._df.columns)[0]
        elem = hv.Histogram(self._df, col, bins=bins, range=range)
        return _apply_opts(elem, opts)

    def bar(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        **opts: Any,
    ) -> hv.Bars:
        """Vertical bar chart (ideal for vkpandas groupby outputs)."""
        rx, ry = self._resolve_xy(x, y)
        return _apply_opts(hv.Bars(self._df, rx, ry), opts)

    def line(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        **opts: Any,
    ) -> hv.Curve:
        """2D line / time-series curve."""
        rx, ry = self._resolve_xy(x, y)
        return _apply_opts(hv.Curve(self._df, rx, ry), opts)

    def plot3d(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        kind: str = "lines",
        color_by: Optional[str] = None,
        color_key: Any = None,
        basemap: Optional[str] = "esri",
        interact: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Zero-copy Vulkan GPU 3D perspective visualization (static or interactive orbit)."""
        rx, ry = self._resolve_xy(x, y)
        cols = list(getattr(self._df, "columns", []))
        rz = z or (cols[2] if len(cols) >= 3 else "z")
        if interact:
            return vks.interact_3d(
                self._df, rx, ry, rz,
                kind=kind,
                basemap=basemap,
                **kwargs,
            )
        return vks.render_3d(
            self._df, rx, ry, rz,
            kind=kind,
            color_by=color_by,
            color_key=color_key,
            basemap=basemap,
            **kwargs,
        )
