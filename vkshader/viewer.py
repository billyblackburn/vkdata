"""Fast inline Jupyter interactive pan/zoom/shade viewer (~15-24 ms JPEG streaming)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from .canvas import Canvas, SpatialPoints, _extract_gpu_col, shade


def interact(
    cvs: Canvas,
    source: Any,
    x: str,
    y: str,
    how: str = "eq_hist",
    basemap: Optional[str] = "esri",
    alpha: int = 235,
    jpeg_quality: int = 85,
    layers: Optional[Dict[str, Tuple[str, str]]] = None,
):
    """Display an interactive GPU pan/zoom/shade viewer inline in Jupyter (mouse drag, wheel, box-zoom)."""
    import ipywidgets as widgets
    from IPython.display import display

    if cvs.x_range is None or cvs.y_range is None:
        raise ValueError("Canvas must have x_range and y_range set for interact()")

    layer_map: Dict[str, Tuple[str, str]] = layers or {"default": (x, y)}
    first_layer = next(iter(layer_map))
    spatial_cache: Dict[str, SpatialPoints] = {}

    def _get_spatial(layer_name: str) -> SpatialPoints:
        if layer_name not in spatial_cache:
            cx, cy = layer_map[layer_name]
            spatial_cache[layer_name] = SpatialPoints(_extract_gpu_col(source, cx), _extract_gpu_col(source, cy))
        return spatial_cache[layer_name]

    init_xr = (float(cvs.x_range[0]), float(cvs.x_range[1]))
    init_yr = (float(cvs.y_range[0]), float(cvs.y_range[1]))
    state = {"xr": init_xr, "yr": init_yr, "how": how, "layer": first_layer, "spread": -1}

    def _render() -> Tuple[bytes, float, float]:
        t0 = time.perf_counter()
        sp = _get_spatial(state["layer"])
        agg = cvs.points(sp, x_range=state["xr"], y_range=state["yr"])
        img = shade(agg, how=state["how"], alpha=alpha, basemap=basemap, spread_px=state["spread"])
        gpu_ms = (time.perf_counter() - t0) * 1000.0
        jpg = img.to_jpeg(quality=jpeg_quality)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return jpg, gpu_ms, total_ms

    jpg0, g0, tot0 = _render()
    disp_w = min(720, cvs.plot_width)
    disp_h = int(round(disp_w * (cvs.plot_height / max(1, cvs.plot_width))))
    status = widgets.Label(
        value=f"how={state['how']} | spread=auto | GPU={g0:.1f} ms | total={tot0:.1f} ms"
    )

    use_anywidget = True
    try:
        from .widget import InteractiveMapWidget
    except ImportError:
        use_anywidget = False

    if use_anywidget:
        map_w = None

        def _on_vp(xr: Tuple[float, float], yr: Tuple[float, float]) -> None:
            state["xr"], state["yr"] = xr, yr
            jpg, g_ms, tot_ms = _render()
            assert map_w is not None
            map_w.image_bytes = jpg
            sp_lbl = "auto" if state["spread"] < 0 else str(state["spread"])
            status.value = (
                f"layer={state['layer']} | how={state['how']} | spread={sp_lbl} | "
                f"GPU={g_ms:.1f} ms | total={tot_ms:.1f} ms"
            )

        map_w = InteractiveMapWidget(
            width=disp_w,
            height=disp_h,
            x_range=init_xr,
            y_range=init_yr,
            image_bytes=jpg0,
            on_viewport_change=_on_vp,
        )

        def _refresh() -> None:
            jpg, g_ms, tot_ms = _render()
            map_w.image_bytes = jpg
            sp_lbl = "auto" if state["spread"] < 0 else str(state["spread"])
            status.value = (
                f"layer={state['layer']} | how={state['how']} | spread={sp_lbl} | "
                f"GPU={g_ms:.1f} ms | total={tot_ms:.1f} ms"
            )

        def _on_reset(_: Any) -> None:
            state["xr"], state["yr"] = init_xr, init_yr
            map_w.viewport = [init_xr[0], init_xr[1], init_yr[0], init_yr[1]]

        btn_reset = widgets.Button(description="Reset", button_style="primary")
        btn_reset.on_click(_on_reset)
        how_dd = widgets.Dropdown(options=["eq_hist", "log", "cbrt", "linear"], value=how, description="Shade:")
        spread_dd = widgets.Dropdown(
            options=[("auto (dynspread)", -1), ("0 (off)", 0), ("1 (3x3)", 1), ("2 (5x5)", 2)],
            value=-1,
            description="Spread:",
        )
        how_dd.observe(lambda ch: (state.update({"how": ch["new"]}), _refresh()), names="value")
        spread_dd.observe(lambda ch: (state.update({"spread": ch["new"]}), _refresh()), names="value")

        controls = [btn_reset, how_dd, spread_dd]
        if len(layer_map) > 1:
            layer_dd = widgets.Dropdown(options=list(layer_map.keys()), value=first_layer, description="Layer:")
            layer_dd.observe(lambda ch: (state.update({"layer": ch["new"]}), _refresh()), names="value")
            controls.insert(1, layer_dd)

        box = widgets.VBox([widgets.HBox(controls), status, map_w])
        display(box)
        return box

    # Fallback button-based viewer if anywidget is unavailable
    img_widget = widgets.Image(value=jpg0, format="jpeg", layout=widgets.Layout(width=f"{disp_w}px"))
    btn_in = widgets.Button(description="Zoom +", button_style="primary")
    btn_out = widgets.Button(description="Zoom -")
    btn_reset = widgets.Button(description="Reset")

    def _refresh_fb() -> None:
        jpg, g_ms, tot_ms = _render()
        img_widget.value = jpg
        status.value = f"how={state['how']} | GPU={g_ms:.1f} ms | total={tot_ms:.1f} ms"

    def _zoom_fb(scale: float) -> None:
        xr0, xr1 = state["xr"]
        yr0, yr1 = state["yr"]
        cx, cy = 0.5 * (xr0 + xr1), 0.5 * (yr0 + yr1)
        hx, hy = 0.5 * (xr1 - xr0) * scale, 0.5 * (yr1 - yr0) * scale
        state["xr"], state["yr"] = (cx - hx, cx + hx), (cy - hy, cy + hy)
        _refresh_fb()

    btn_in.on_click(lambda _: _zoom_fb(0.5))
    btn_out.on_click(lambda _: _zoom_fb(2.0))
    btn_reset.on_click(lambda _: (state.update({"xr": init_xr, "yr": init_yr}), _refresh_fb()))
    box = widgets.VBox([widgets.HBox([btn_in, btn_out, btn_reset]), status, img_widget])
    display(box)
    return box
