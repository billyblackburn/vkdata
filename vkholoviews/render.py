"""Rendering and interactive Jupyter viewer for vkholoviews."""

from __future__ import annotations

import math
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import vknumpy as vnp
import vkshader as vks

from .core import Layout, Overlay, Viewable
from .elements import Bars, Curve, Histogram, Path, Points, Tiles, _col_to_vk
from .operation import DatashadedPaths, DatashadedPoints

_BG_DARK = (18, 20, 26, 255)
_PLOT_BG = (24, 27, 36, 255)
_GRID_COLOR = (46, 52, 66, 255)
_AXIS_COLOR = (130, 140, 160, 255)
_TEXT_COLOR = (225, 230, 240, 255)
_ACCENT = (255, 145, 40, 255)
_ACCENT_LINE = (80, 195, 255, 255)


def _col_to_np(source: Any, col: str) -> np.ndarray:
    c = source[col]
    if hasattr(c, "to_numpy"):
        return c.to_numpy()
    return np.asarray(c)


def _fmt_num(v: float) -> str:
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:.1f}M"
    if av >= 1e3:
        return f"{v / 1e3:.1f}k"
    if av >= 100 or v == int(v):
        return f"{v:.0f}"
    return f"{v:.1f}"


def _draw_frame(
    width: int,
    height: int,
    title: str,
    xlabel: str,
    ylabel: str,
    x_ticks: List[Tuple[float, str]],
    y_ticks: List[Tuple[float, str]],
) -> Tuple[Image.Image, ImageDraw.ImageDraw, Tuple[int, int, int, int]]:
    img = Image.new("RGBA", (width, height), _BG_DARK)
    draw = ImageDraw.Draw(img)

    pad_l, pad_r = 62, 18
    pad_t = 32 if title else 16
    pad_b = 42
    x0, y0, x1, y1 = pad_l, pad_t, width - pad_r, height - pad_b

    draw.rectangle([x0, y0, x1, y1], fill=_PLOT_BG, outline=_AXIS_COLOR, width=1)

    if title:
        draw.text((x0, 10), title, fill=_TEXT_COLOR)
    if xlabel:
        draw.text(((x0 + x1) // 2 - len(xlabel) * 3, height - 16), xlabel, fill=_AXIS_COLOR)
    if ylabel:
        draw.text((8, 10), ylabel, fill=_AXIS_COLOR)

    for frac, lbl in y_ticks:
        py = int(round(y1 - frac * (y1 - y0)))
        if y0 < py < y1:
            draw.line([(x0 + 1, py), (x1 - 1, py)], fill=_GRID_COLOR, width=1)
        draw.text((8, py - 5), lbl, fill=_AXIS_COLOR)

    for frac, lbl in x_ticks:
        px = int(round(x0 + frac * (x1 - x0)))
        if x0 < px < x1:
            draw.line([(px, y0 + 1), (px, y1 - 1)], fill=_GRID_COLOR, width=1)
        draw.text((px - len(lbl) * 3, y1 + 6), lbl, fill=_AXIS_COLOR)

    return img, draw, (x0, y0, x1, y1)


def _render_bars(elem: Bars) -> Image.Image:
    w = int(elem.get_opt("width", 460))
    h = int(elem.get_opt("height", 320))
    title = elem.get_opt("title", f"{elem.vdims[0]} by {elem.kdims[0]}")
    color = elem.get_opt("color", _ACCENT)

    xs = _col_to_np(elem.data, elem.kdims[0])
    ys = _col_to_np(elem.data, elem.vdims[0]).astype(np.float64)
    n = len(xs)
    ymax = max(1e-6, float(np.nanmax(ys)) * 1.1) if n > 0 else 1.0

    y_ticks = [(i / 4.0, _fmt_num(ymax * i / 4.0)) for i in range(5)]
    step = max(1, n // 6)
    x_ticks = [((i + 0.5) / max(1, n), str(xs[i])) for i in range(0, n, step)]

    img, draw, (x0, y0, x1, y1) = _draw_frame(w, h, title, elem.kdims[0], elem.vdims[0], x_ticks, y_ticks)
    if n == 0:
        return img

    plot_w = x1 - x0
    plot_h = y1 - y0
    bw = plot_w / n
    gap = max(1.0, bw * 0.15)
    for i, val in enumerate(ys):
        if np.isnan(val):
            continue
        bx0 = int(round(x0 + i * bw + gap * 0.5))
        bx1 = int(round(x0 + (i + 1) * bw - gap * 0.5))
        by0 = min(y1 - 1, int(round(y1 - max(0.0, val / ymax) * plot_h)))
        draw.rectangle([bx0, by0, max(bx0 + 1, bx1), y1 - 1], fill=color)
    return img


def _render_histogram(
    elem: Histogram,
    counts_edges: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title_override: Optional[str] = None,
) -> Image.Image:
    w = int(elem.get_opt("width", 460))
    h = int(elem.get_opt("height", 320))
    title = title_override if title_override is not None else elem.get_opt("title", f"Histogram: {elem.kdims[0]}")
    color = elem.get_opt("color", _ACCENT_LINE)

    counts, edges = counts_edges if counts_edges is not None else elem.compute_gpu()
    n = len(counts)
    ymax = max(1.0, float(counts.max()) * 1.1) if n > 0 else 1.0

    y_ticks = [(i / 4.0, _fmt_num(ymax * i / 4.0)) for i in range(5)]
    x_ticks = [(i / 4.0, _fmt_num(float(edges[0] + (edges[-1] - edges[0]) * i / 4.0))) for i in range(5)]

    img, draw, (x0, y0, x1, y1) = _draw_frame(w, h, title, elem.kdims[0], "count", x_ticks, y_ticks)
    plot_w = x1 - x0
    plot_h = y1 - y0
    bw = plot_w / max(1, n)
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        bx0 = int(round(x0 + i * bw))
        bx1 = int(round(x0 + (i + 1) * bw)) - 1
        by0 = min(y1 - 1, int(round(y1 - (c / ymax) * plot_h)))
        draw.rectangle([bx0, by0, max(bx0 + 1, bx1), y1 - 1], fill=color)
    return img


def _render_curve(elem: Curve) -> Image.Image:
    w = int(elem.get_opt("width", 460))
    h = int(elem.get_opt("height", 320))
    title = elem.get_opt("title", f"{elem.vdims[0]} vs {elem.kdims[0]}")
    color = elem.get_opt("color", _ACCENT_LINE)

    xs = _col_to_np(elem.data, elem.kdims[0]).astype(np.float64)
    ys = _col_to_np(elem.data, elem.vdims[0]).astype(np.float64)
    n = len(xs)
    if n == 0:
        img, _, _ = _draw_frame(w, h, title, elem.kdims[0], elem.vdims[0], [], [])
        return img

    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad_y = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad_y, ymax + pad_y

    y_ticks = [(i / 4.0, _fmt_num(ymin + (ymax - ymin) * i / 4.0)) for i in range(5)]
    x_ticks = [(i / 4.0, _fmt_num(xmin + (xmax - xmin) * i / 4.0)) for i in range(5)]

    img, draw, (x0, y0, x1, y1) = _draw_frame(w, h, title, elem.kdims[0], elem.vdims[0], x_ticks, y_ticks)
    pts = []
    for xv, yv in zip(xs, ys):
        if np.isnan(xv) or np.isnan(yv):
            continue
        px = int(round(x0 + (xv - xmin) / (xmax - xmin) * (x1 - x0)))
        py = int(round(y1 - (yv - ymin) / (ymax - ymin) * (y1 - y0)))
        pts.append((px, py))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=2)
    for px, py in pts:
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
    return img


def _infer_ranges(ds: DatashadedPoints, opts: dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xr = opts.get("xlim") or ds.get_opt("xlim")
    yr = opts.get("ylim") or ds.get_opt("ylim")
    if xr is None:
        xs = _col_to_vk(ds.data, ds.kdims[0])
        xr = (float(vnp.nanmin(xs).item()), float(vnp.nanmax(xs).item()))
    if yr is None:
        ys = _col_to_vk(ds.data, ds.kdims[1])
        yr = (float(vnp.nanmin(ys).item()), float(vnp.nanmax(ys).item()))
    return (float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1]))


def _render_spatial_overlay(layers: List[Viewable], opts: dict) -> Image.Image:
    w = int(opts.get("width", 640))
    h = int(opts.get("height", 640))
    title = opts.get("title", "")

    basemap: Optional[str] = None
    ds_layers: List[Any] = []
    curve_layers: List[Curve] = []
    poly_layers: List[Any] = []
    for lyr in layers:
        if isinstance(lyr, Tiles):
            basemap = lyr.source
        elif isinstance(lyr, (DatashadedPoints, DatashadedPaths)):
            ds_layers.append(lyr)
        elif isinstance(lyr, Points):
            ds_layers.append(DatashadedPoints(lyr, how=lyr.get_opt("how", opts.get("how", "eq_hist"))))
        elif isinstance(lyr, Path):
            ds_layers.append(DatashadedPaths(lyr, how=lyr.get_opt("how", opts.get("how", "eq_hist"))))
        elif isinstance(lyr, Curve):
            curve_layers.append(lyr)
        elif hasattr(lyr, "draw_onto") and hasattr(lyr, "bounds_mercator"):
            poly_layers.append(lyr)

    if not ds_layers and not poly_layers and not curve_layers:
        return Image.new("RGBA", (w, h), _BG_DARK)

    if ds_layers:
        xr, yr = _infer_ranges(ds_layers[0], opts)
        first = ds_layers[0]
        shaded = first.render_gpu(
            w, h, xr, yr,
            basemap=basemap,
            how=first.get_opt("how") or opts.get("how"),
            cmap=first.get_opt("cmap") or opts.get("cmap"),
        )
        pil_img = shaded.to_pil().convert("RGBA")
        for extra in ds_layers[1:]:
            ex_shaded = extra.render_gpu(
                w, h, xr, yr,
                basemap=None,
                how=extra.get_opt("how") or opts.get("how"),
                cmap=extra.get_opt("cmap") or opts.get("cmap"),
            )
            ex_arr = np.array(ex_shaded.to_pil().convert("RGBA"), copy=True)
            bg_mask = (ex_arr[:, :, 0] == 10) & (ex_arr[:, :, 1] == 10) & (ex_arr[:, :, 2] == 14)
            ex_arr[bg_mask, 3] = 0
            pil_img = Image.alpha_composite(pil_img, Image.fromarray(ex_arr, mode="RGBA"))
    else:
        xr = opts.get("xlim") or (poly_layers[0].get_opt("xlim") if poly_layers else None)
        yr = opts.get("ylim") or (poly_layers[0].get_opt("ylim") if poly_layers else None)
        if (xr is None or yr is None) and poly_layers:
            bxr, byr = poly_layers[0].bounds_mercator()
            xr = xr or bxr
            yr = yr or byr
        xr = (float(xr[0]), float(xr[1])) if xr is not None else (0.0, 1.0)
        yr = (float(yr[0]), float(yr[1])) if yr is not None else (0.0, 1.0)
        if basemap:
            empty_agg = vks.AggResult(grid=vnp.zeros((h, w), dtype=np.int32), x_range=xr, y_range=yr)
            pil_img = vks.shade(empty_agg, basemap=basemap, spread_px=0).to_pil().convert("RGBA")
        else:
            pil_img = Image.new("RGBA", (w, h), _BG_DARK)

    for plyr in poly_layers:
        pil_img = plyr.draw_onto(pil_img, xr, yr)

    if curve_layers:
        draw_c = ImageDraw.Draw(pil_img)
        x_span = max(xr[1] - xr[0], 1e-12)
        y_span = max(yr[1] - yr[0], 1e-12)
        for clyr in curve_layers:
            cxs = _col_to_np(clyr.data, clyr.kdims[0]).astype(np.float64)
            cys = _col_to_np(clyr.data, clyr.vdims[0]).astype(np.float64)
            color = clyr.get_opt("color", _ACCENT_LINE)
            lw = int(clyr.get_opt("line_width", 2))
            seg = []
            for xv, yv in zip(cxs, cys):
                if np.isnan(xv) or np.isnan(yv):
                    if len(seg) >= 2:
                        draw_c.line(seg, fill=color, width=lw)
                    seg = []
                    continue
                px = int(round((xv - xr[0]) / x_span * (w - 1)))
                py = int(round((yr[1] - yv) / y_span * (h - 1)))
                seg.append((px, py))
            if len(seg) >= 2:
                draw_c.line(seg, fill=color, width=lw)

    if title:
        canvas = Image.new("RGBA", (w, h + 28), _BG_DARK)
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 7), title, fill=_TEXT_COLOR)
        canvas.paste(pil_img, (0, 28))
        return canvas
    return pil_img


def _render_layout(layout: Layout) -> Image.Image:
    imgs = [render_viewable(it) for it in layout.items]
    if not imgs:
        return Image.new("RGBA", (320, 240), _BG_DARK)

    n = len(imgs)
    cols = layout._cols if layout._cols is not None else min(n, 2 if n > 2 else n)
    rows = int(math.ceil(n / cols))
    pad = 12

    col_w = [0] * cols
    row_h = [0] * rows
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        col_w[c] = max(col_w[c], im.width)
        row_h[r] = max(row_h[r], im.height)

    total_w = sum(col_w) + pad * (cols + 1)
    total_h = sum(row_h) + pad * (rows + 1)
    canvas = Image.new("RGBA", (total_w, total_h), _BG_DARK)

    y = pad
    for r in range(rows):
        x = pad
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                canvas.paste(imgs[idx], (x, y))
            x += col_w[c] + pad
        y += row_h[r] + pad
    return canvas


def render_viewable(obj: Viewable) -> Image.Image:
    if isinstance(obj, Layout):
        return _render_layout(obj)
    if isinstance(obj, Overlay):
        return _render_spatial_overlay(obj.layers, obj._opts)
    if isinstance(obj, (DatashadedPoints, Points, DatashadedPaths, Path)) or (
        hasattr(obj, "draw_onto") and hasattr(obj, "bounds_mercator")
    ):
        return _render_spatial_overlay([obj], obj._opts)
    if isinstance(obj, Bars):
        return _render_bars(obj)
    if isinstance(obj, Histogram):
        return _render_histogram(obj)
    if isinstance(obj, Curve):
        return _render_curve(obj)
    if isinstance(obj, Tiles):
        return Image.new("RGBA", (int(obj.get_opt("width", 400)), int(obj.get_opt("height", 400))), _BG_DARK)
    raise TypeError(f"Unsupported Viewable: {type(obj).__name__}")


def interact_viewable(obj: Viewable, jpeg_quality: int = 85) -> Any:
    """Interactive GPU pan/zoom/shade/dynspread Jupyter widget with live linked cross-filtering!

    Works on:
      - A spatial Overlay (`Tiles * datashade(Points)`)
      - A linked Layout (`(Tiles * datashade(Points)) + Histogram(df, 'fare_amount')`)
        where panning/zooming the map live-recomputes the linked Histogram(s) in VRAM (~5 ms)!
    """
    import io
    import ipywidgets as widgets
    from IPython.display import display

    linked_hists: List[Histogram] = []
    spatial_obj: Optional[Viewable] = None

    if isinstance(obj, Layout):
        for it in obj.items:
            if isinstance(it, Histogram):
                linked_hists.append(it)
            elif spatial_obj is None:
                spatial_obj = it
    else:
        spatial_obj = obj

    if spatial_obj is None:
        display(obj)
        return obj

    layers = spatial_obj.layers if isinstance(spatial_obj, Overlay) else [spatial_obj]
    opts = spatial_obj._opts

    basemap: Optional[str] = None
    ds_layer: Optional[Any] = None
    for lyr in layers:
        if isinstance(lyr, Tiles):
            basemap = lyr.source
        elif isinstance(lyr, (DatashadedPoints, DatashadedPaths)):
            ds_layer = lyr
        elif isinstance(lyr, Points):
            ds_layer = DatashadedPoints(lyr, how=opts.get("how", "eq_hist"))
        elif isinstance(lyr, Path):
            ds_layer = DatashadedPaths(lyr, how=opts.get("how", "eq_hist"))

    if ds_layer is None:
        display(obj)
        return obj

    w = int(opts.get("width", ds_layer.get_opt("width", 640)))
    h = int(opts.get("height", ds_layer.get_opt("height", 640)))
    init_xr, init_yr = _infer_ranges(ds_layer, opts)

    is_paths = isinstance(ds_layer, DatashadedPaths)
    sp = None
    if not is_paths:
        # Pre-sort (x, y) + any linked histogram columns once on GPU for instant viewport slicing
        extra_cols = {}
        for h_elem in linked_hists:
            col_name = h_elem.kdims[0]
            if col_name not in extra_cols:
                extra_cols[col_name] = _col_to_vk(h_elem.data, col_name)

        sp = vks.SpatialPoints(
            _col_to_vk(ds_layer.data, ds_layer.kdims[0]),
            _col_to_vk(ds_layer.data, ds_layer.kdims[1]),
            extra=extra_cols if extra_cols else None,
        )
    cvs = vks.Canvas(w, h, x_range=init_xr, y_range=init_yr)

    state = {
        "xr": init_xr,
        "yr": init_yr,
        "how": opts.get("how", ds_layer.how),
        "cmap": opts.get("cmap", getattr(ds_layer, "cmap", "fire")),
        "spread": 0 if is_paths else -1,  # -1 = GPU auto dynspread for points; 0 = crisp lines for paths
    }

    def _render_all() -> Tuple[bytes, List[bytes], float, float]:
        t0 = time.perf_counter()
        xr, yr = state["xr"], state["yr"]
        if is_paths:
            shaded = ds_layer.render_gpu(
                w, h, xr, yr, basemap=basemap, how=state["how"], spread_px=state["spread"], cmap=state["cmap"]
            )
            map_jpg = shaded.to_jpeg(quality=jpeg_quality)
            hist_pngs: List[bytes] = []
            xs = _col_to_vk(ds_layer.data, ds_layer.kdims[0])
            ys = _col_to_vk(ds_layer.data, ds_layer.kdims[1])
            for h_elem in linked_hists:
                col = h_elem.kdims[0]
                counts, edges = h_elem.compute_gpu(xy=(xs, ys), x_range=xr, y_range=yr)
                n_in = int(counts.sum())
                base_title = h_elem.get_opt("title", f"Histogram: {col}")
                im = _render_histogram(h_elem, counts_edges=(counts, edges), title_override=f"{base_title} (n={n_in:,})")
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=88)
                hist_pngs.append(buf.getvalue())
            total_ms = (time.perf_counter() - t0) * 1000.0
            return map_jpg, hist_pngs, total_ms, float(xs.size)

        assert sp is not None
        agg = cvs.points(sp, x_range=xr, y_range=yr)
        shaded = vks.shade(agg, how=state["how"], alpha=ds_layer.alpha, basemap=basemap, spread_px=state["spread"], cmap=state["cmap"])
        map_jpg = shaded.to_jpeg(quality=jpeg_quality)

        hist_pngs = []
        xs, ys = sp.slice_x(xr[0], xr[1])
        for h_elem in linked_hists:
            col = h_elem.kdims[0]
            vs = sp.slice_extra(col, xr[0], xr[1])
            counts, edges = h_elem.compute_gpu(xy=(xs, ys), x_range=xr, y_range=yr, values_override=vs)
            n_in = int(counts.sum())
            base_title = h_elem.get_opt("title", f"Histogram: {col}")
            im = _render_histogram(h_elem, counts_edges=(counts, edges), title_override=f"{base_title} (n={n_in:,})")
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=88)
            hist_pngs.append(buf.getvalue())

        total_ms = (time.perf_counter() - t0) * 1000.0
        return map_jpg, hist_pngs, total_ms, float(xs.size)

    map_jpg0, hist_jpgs0, tot0, _ = _render_all()
    disp_w = min(640, w)
    disp_h = int(round(disp_w * (h / max(1, w))))
    hist_widgets = [
        widgets.Image(value=hj, format="jpeg", layout=widgets.Layout(width=f"{int(h_el.get_opt('width', 440))}px"))
        for hj, h_el in zip(hist_jpgs0, linked_hists)
    ]
    status = widgets.Label(value=f"how={state['how']} | spread=auto | total={tot0:.1f} ms")

    from vkshader.widget import InteractiveMapWidget

    map_widget = None

    def _update_views() -> None:
        m_jpg, h_jpgs, tot_ms, _ = _render_all()
        assert map_widget is not None
        map_widget.image_bytes = m_jpg
        for hw, hj in zip(hist_widgets, h_jpgs):
            hw.value = hj
        sp_lbl = "auto" if state["spread"] < 0 else str(state["spread"])
        status.value = (
            f"how={state['how']} | spread={sp_lbl} | total={tot_ms:.1f} ms | "
            f"x={state['xr'][0]:.0f}..{state['xr'][1]:.0f}"
        )

    def _on_vp(xr: Tuple[float, float], yr: Tuple[float, float]) -> None:
        state["xr"], state["yr"] = xr, yr
        _update_views()

    map_widget = InteractiveMapWidget(
        width=disp_w,
        height=disp_h,
        x_range=init_xr,
        y_range=init_yr,
        image_bytes=map_jpg0,
        on_viewport_change=_on_vp,
    )

    def _on_reset(_: Any) -> None:
        state["xr"], state["yr"] = init_xr, init_yr
        map_widget.viewport = [init_xr[0], init_xr[1], init_yr[0], init_yr[1]]

    btn_reset = widgets.Button(description="Reset", button_style="primary")
    how_dd = widgets.Dropdown(options=["eq_hist", "log", "cbrt", "linear"], value=state["how"], description="Shade:")
    cmap_opts = ["fire", "bmy", "bgy", "kbc", "viridis", "inferno", "plasma", "magma", "turbo", "coolwarm", "rainbow", "gray"]
    cur_cmap = state["cmap"] if isinstance(state["cmap"], str) and state["cmap"] in cmap_opts else "fire"
    cmap_dd = widgets.Dropdown(options=cmap_opts, value=cur_cmap, description="Cmap:")
    spread_dd = widgets.Dropdown(
        options=[("auto (dynspread)", -1), ("0 (off)", 0), ("1 (3x3)", 1), ("2 (5x5)", 2)],
        value=state["spread"],
        description="Spread:",
    )

    btn_reset.on_click(_on_reset)
    how_dd.observe(lambda ch: (state.update({"how": ch["new"]}), _update_views()), names="value")
    cmap_dd.observe(lambda ch: (state.update({"cmap": ch["new"]}), _update_views()), names="value")
    spread_dd.observe(lambda ch: (state.update({"spread": ch["new"]}), _update_views()), names="value")

    toolbar = widgets.HBox([btn_reset, how_dd, cmap_dd, spread_dd])
    body = widgets.HBox([map_widget, widgets.VBox(hist_widgets)]) if hist_widgets else map_widget
    box = widgets.VBox([toolbar, status, body])
    display(box)
    return box
