"""vkplot.charts: Reusable SynthWave '84 one-liner ML & Time-Series charts (zero matplotlib dependency)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

import vkcolorcet as vkc
import vkpandas as vpd
import vkshader as vks


_DEFAULT_COLORS = ["#36f9f6", "#ff7edb", "#ff9e64", "#72f1b8", "#fede5d", "#b6b1d6"]


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "_columns") and hasattr(x, "columns"):
        return np.column_stack([x[c].to_numpy() for c in x.columns]).astype(np.float32)
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), dtype=np.float32)
    if hasattr(x, "numpy") and callable(x.numpy):
        return np.asarray(x.numpy(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def side_by_side(*images: Image.Image, pad: int = 16, bg: str = "#1b1728") -> Image.Image:
    """Compose two or more PIL images horizontally onto a clean SynthWave '84 card."""
    imgs = [im.convert("RGB") for im in images]
    total_w = sum(im.width for im in imgs) + pad * (len(imgs) + 1)
    max_h = max(im.height for im in imgs) + pad * 2
    canvas = Image.new("RGB", (total_w, max_h), bg)
    x = pad
    for im in imgs:
        y = pad + (max_h - 2 * pad - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + pad
    return canvas


def line_chart(
    series: Sequence[Tuple[Any, ...]],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    vline: Optional[Tuple[float, str]] = None,
    hlines: Optional[Sequence[Tuple[float, str, str]]] = None,
    width: int = 900,
    height: int = 340,
    y_min: Optional[float] = None,
) -> Image.Image:
    """Render one or more 2D line series on a crisp SynthWave '84 dark canvas.

    Each entry in ``series`` may be ``(xs, ys, label)``, ``(xs, ys, color, label)``,
    or ``(xs, ys, color, line_width, label)``.
    """
    parsed = []
    for idx, item in enumerate(series):
        if len(item) == 2:
            xs, ys = _to_np(item[0]).ravel(), _to_np(item[1]).ravel()
            col, lw, lbl = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)], 2, ""
        elif len(item) == 3:
            xs, ys = _to_np(item[0]).ravel(), _to_np(item[1]).ravel()
            col, lw, lbl = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)], 2, str(item[2])
        elif len(item) == 4:
            xs, ys = _to_np(item[0]).ravel(), _to_np(item[1]).ravel()
            col, lw, lbl = str(item[2]), 2, str(item[3])
        else:
            xs, ys = _to_np(item[0]).ravel(), _to_np(item[1]).ravel()
            col, lw, lbl = str(item[2]), int(item[3]), str(item[4])
        parsed.append((xs, ys, col, lw, lbl))

    img = Image.new("RGB", (width, height), "#1b1728")
    d = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 58, 22, 42, 40
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    d.rectangle([pad_l, pad_t, width - pad_r, height - pad_b], fill="#171324", outline="#495495", width=1)

    all_x = np.concatenate([p[0] for p in parsed])
    all_y = np.concatenate([p[1] for p in parsed])
    xmin, xmax = float(all_x.min()), float(all_x.max())
    h_vals = [float(h[0]) for h in hlines] if hlines else []
    ymin = float(y_min) if y_min is not None else min([float(all_y.min()), *h_vals])
    ymax = max([float(all_y.max()), *h_vals])
    if ymax - ymin < 1e-6:
        ymax = ymin + 1.0
    if y_min is None:
        ymin -= 0.04 * (ymax - ymin)
    ymax += 0.08 * (ymax - ymin)

    def to_px(xv: float, yv: float) -> Tuple[float, float]:
        px = pad_l + (xv - xmin) / (xmax - xmin + 1e-9) * pw
        py = pad_t + ph - (yv - ymin) / (ymax - ymin + 1e-9) * ph
        return (float(px), float(py))

    for g in range(5):
        gy = pad_t + g * ph / 4
        val = ymax - g * (ymax - ymin) / 4
        d.line([(pad_l, gy), (width - pad_r, gy)], fill="#2a2342", width=1)
        d.text((8, gy - 6), f"{val:5.1f}", fill="#848bbd")
    for g in range(6):
        gx = pad_l + g * pw / 5
        xv = xmin + g * (xmax - xmin) / 5
        d.line([(gx, pad_t), (gx, height - pad_b)], fill="#241e38", width=1)
        fmt = f"{xv:.2f}" if (xmax - xmin) <= 2.0 else f"{xv:.0f}"
        d.text((gx - 14, height - pad_b + 8), fmt, fill="#848bbd")

    if vline is not None:
        vx_val, v_lbl = float(vline[0]), str(vline[1])
        vx, _ = to_px(vx_val, ymin)
        d.line([(vx, pad_t), (vx, height - pad_b)], fill="#ff7edb", width=2)
        if v_lbl:
            d.text((vx + 6, pad_t + 6), v_lbl, fill="#ff7edb")

    if hlines:
        for h_val, h_lbl, h_col in hlines:
            _, hy = to_px(xmin, float(h_val))
            for x0 in range(pad_l, width - pad_r, 14):
                d.line([(x0, hy), (min(x0 + 7, width - pad_r), hy)], fill=h_col, width=1)
            if h_lbl:
                d.text((width - pad_r - len(h_lbl) * 6 - 10, max(pad_t + 2, hy - 14)), h_lbl, fill=h_col)

    for xs, ys, col, lw, _ in parsed:
        pts = [to_px(float(x), float(y)) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=lw)

    if title:
        d.text((pad_l, 12), title, fill="#f4eee4")
    if x_label:
        d.text((width // 2 - len(x_label) * 3, height - 16), x_label, fill="#b6b1d6")
    lx = width - pad_r - 10
    for _, _, col, _, lbl in reversed(parsed):
        if not lbl:
            continue
        lx -= len(lbl) * 6 + 22
        d.rectangle([lx, 14, lx + 10, 22], fill=col)
        d.text((lx + 14, 12), lbl, fill="#f4eee4")
    return img


def barh_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "",
    positive_color: str = "#72f1b8",
    negative_color: str = "#ff7edb",
    width: int = 760,
    height: Optional[int] = None,
) -> Image.Image:
    """Render a horizontal bar chart (ideal for feature correlations or importances)."""
    vals = [float(v) for v in _to_np(values).ravel()]
    lbls = [str(l) for l in labels]
    n = len(lbls)
    h = int(height) if height is not None else max(180, 56 + n * 24)
    img = Image.new("RGB", (width, h), "#1b1728")
    dr = ImageDraw.Draw(img)
    if title:
        dr.text((16, 12), title, fill="#f4eee4")

    has_neg = any(v < 0 for v in vals)
    max_abs = max((abs(v) for v in vals), default=1.0) or 1.0
    label_w = min(280, max((len(l) for l in lbls), default=10) * 7 + 24)
    if has_neg:
        x_zero = label_w + (width - label_w - 64) // 2
        scale = (width - label_w - 110) / (2.0 * max_abs)
    else:
        x_zero = label_w
        scale = (width - label_w - 76) / max_abs

    dr.line([(x_zero, 38), (x_zero, h - 16)], fill="#495495", width=1)
    for i, (lbl, v) in enumerate(zip(lbls, vals)):
        y0 = 44 + i * 24
        dr.text((16, y0 + 2), lbl, fill="#ded7ee")
        x1 = x_zero + int(v * scale)
        col = positive_color if v >= 0 else negative_color
        dr.rectangle([min(x_zero, x1), y0 + 2, max(x_zero, x1), y0 + 17], fill=col)
        tx = max(x_zero, x1) + 6 if v >= 0 else min(x_zero, x1) - 48
        dr.text((tx, y0 + 2), f"{v:+.3f}" if has_neg else f"{v:.3f}", fill="#f4eee4")
    return img


def pred_vs_actual(
    y_true: Any,
    y_pred: Any,
    *,
    title: str = "Predicted vs. Actual",
    cmap: str = "fire",
    lim: Optional[Tuple[float, float]] = None,
    width: int = 380,
    height: int = 340,
) -> Image.Image:
    """Render a GPU-datashaded Predicted vs. Actual scatter plot with a perfect y=x diagonal."""
    yt = _to_np(y_true).ravel()
    yp = _to_np(y_pred).ravel()
    lo, hi = lim if lim is not None else (float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max())))
    yp_c = np.clip(yp, lo, hi).astype(np.float32)

    cvs = vks.Canvas(plot_width=width, plot_height=height, x_range=(lo, hi), y_range=(lo, hi))
    agg = cvs.points(vpd.DataFrame({"yt": yt, "yp": yp_c}), "yt", "yp")
    im = vks.shade(agg, cmap=cmap, how="eq_hist", spread_px=1).to_pil().convert("RGB")
    dr = ImageDraw.Draw(im)
    for step in range(0, min(width, height), 10):
        x0 = int(step * width / min(width, height))
        y0 = height - 1 - int(step * height / min(width, height))
        dr.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill="#36f9f6")

    card = Image.new("RGB", (width + 16, height + 40), "#1b1728")
    card.paste(im, (8, 32))
    cdr = ImageDraw.Draw(card)
    cdr.text((12, 10), title, fill="#f4eee4")
    return card


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str],
    *,
    title: str = "Confusion Matrix (rows = True, cols = Pred)",
    cell_px: int = 44,
) -> Image.Image:
    """Render a SynthWave-colored Confusion Matrix heatmap with per-cell counts."""
    yt = _to_np(y_true).astype(np.int32).ravel()
    yp = _to_np(y_pred).astype(np.int32).ravel()
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int32)
    for a, b in zip(yt, yp):
        if 0 <= a < k and 0 <= b < k:
            cm[a, b] += 1

    lut = vkc.get_rgb_lut("fire")
    pad_l, pad_t, pad_r, pad_b = 110, 56, 20, 22
    W = pad_l + k * cell_px + pad_r
    H = pad_t + k * cell_px + pad_b
    img = Image.new("RGB", (W, H), "#1b1728")
    dr = ImageDraw.Draw(img)
    dr.text((14, 10), title, fill="#f4eee4")

    row_max = np.maximum(cm.max(axis=1, keepdims=True), 1)
    for r in range(k):
        dr.text((10, pad_t + r * cell_px + cell_px // 3), str(class_names[r])[:13], fill="#ded7ee")
        dr.text((pad_l + r * cell_px + 4, pad_t - 18), str(class_names[r])[:5], fill="#36f9f6")
        for c in range(k):
            v = int(cm[r, c])
            norm = min(1.0, v / float(row_max[r, 0]))
            rgb = tuple(int(x) for x in lut[int(norm * 255)])
            x0, y0 = pad_l + c * cell_px, pad_t + r * cell_px
            dr.rectangle([x0 + 1, y0 + 1, x0 + cell_px - 2, y0 + cell_px - 2], fill=rgb)
            txt_col = "#1b1728" if norm > 0.62 else "#f4eee4"
            dr.text((x0 + 8, y0 + cell_px // 3), str(v), fill=txt_col)
    return img


def image_grid(
    images: Sequence[Any],
    *,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 5,
    scale: int = 3,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    pad: int = 10,
) -> Image.Image:
    """Arrange 2D numpy/GPU images (e.g. Fashion-MNIST prototypes, PCA Eigen-Clothes, CNN filters) into a grid."""
    arrs = [_to_np(im) for im in images]
    n = len(arrs)
    if n == 0:
        return Image.new("RGB", (200, 60), "#1b1728")
    h0, w0 = arrs[0].shape[:2]
    cw, ch = int(w0 * scale), int(h0 * scale)
    nrows = (n + ncols - 1) // ncols
    lbl_h = 18 if titles is not None else 0
    top_h = 32 if title else 8

    lut = vkc.get_rgb_lut(cmap) if cmap else None
    W = pad + ncols * (cw + pad)
    H = top_h + pad + nrows * (ch + lbl_h + pad)
    canvas = Image.new("RGB", (W, H), "#1b1728")
    dr = ImageDraw.Draw(canvas)
    if title:
        dr.text((pad, 8), title, fill="#f4eee4")

    for i, a in enumerate(arrs):
        r, c = divmod(i, ncols)
        x0 = pad + c * (cw + pad)
        y0 = top_h + pad + r * (ch + lbl_h + pad)
        lo, hi = float(a.min()), float(a.max())
        norm = np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)
        if lut is not None:
            tile = Image.fromarray(lut[u8], mode="RGB")
        elif a.ndim == 2:
            tile = Image.fromarray(u8, mode="L").convert("RGB")
        else:
            tile = Image.fromarray(u8, mode="RGB")
        tile = tile.resize((cw, ch), Image.Resampling.NEAREST)
        canvas.paste(tile, (x0, y0))
        if titles is not None and i < len(titles):
            dr.text((x0, y0 + ch + 3), str(titles[i])[:14], fill="#ded7ee")
    return canvas


def decision_boundaries(
    panels: Sequence[Tuple[str, Any]],
    X: Any,
    y: Any,
    *,
    bounds: Tuple[float, float] = (-2.3, 2.3),
    cmap: str = "coolwarm",
    panel_size: int = 240,
) -> Image.Image:
    """Render side-by-side 2D probability/decision-boundary maps with scatter points overlaid."""
    X_np = _to_np(X)
    y_np = _to_np(y).reshape(-1)
    lo, hi = float(bounds[0]), float(bounds[1])
    span = max(hi - lo, 1e-6)
    lut = vkc.get_rgb_lut(cmap)
    pw = ph = int(panel_size)
    cards = []
    for title, probs in panels:
        p2d = _to_np(probs).reshape(ph, pw)
        idx = np.clip(p2d * 255.0, 0, 255).astype(np.int32)
        rgb = lut[idx].copy()
        im = Image.fromarray(rgb, mode="RGB")
        dr = ImageDraw.Draw(im)
        for xi, yi in zip(X_np, y_np):
            px = int((float(xi[0]) - lo) / span * (pw - 1))
            py = int((hi - float(xi[1])) / span * (ph - 1))
            col = (255, 195, 85) if int(round(float(yi))) == 1 else (75, 195, 255)
            dr.ellipse((px - 3, py - 3, px + 3, py + 3), fill=col, outline=(24, 20, 36))
        cards.append((im, title))
    strip = Image.new("RGB", (len(cards) * (pw + 12) + 12, ph + 44), (27, 23, 40))
    dr_s = ImageDraw.Draw(strip)
    for i, (im, ttl) in enumerate(cards):
        x0 = 12 + i * (pw + 12)
        strip.paste(im, (x0, 34))
        dr_s.text((x0 + 6, 10), str(ttl), fill=(244, 238, 228))
    return strip


def class_gallery(
    images: Any,
    labels: Any,
    class_names: Sequence[str],
    *,
    n_samples: int = 8,
    show_mean: bool = True,
    scale: int = 2,
    title: str = "Samples per Class vs. GPU Mean Prototype (far right)",
) -> Image.Image:
    """Render a rows-by-class image gallery with an optional cyan-tinted mean prototype column."""
    imgs = _to_np(images)
    ys = _to_np(labels).reshape(-1).astype(np.int32)
    h0, w0 = imgs.shape[1], imgs.shape[2]
    tile_h, tile_w = int(h0 * scale), int(w0 * scale)
    pad = 4
    left_lbl = 96
    n_cols = n_samples + (1 if show_mean else 0)
    W = left_lbl + n_cols * (tile_w + pad) + 18
    H = 42 + len(class_names) * (tile_h + pad) + 8
    canvas = Image.new("RGB", (W, H), (16, 18, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((14, 12), title, fill=(240, 244, 250))

    for c, cname in enumerate(class_names):
        y0 = 42 + c * (tile_h + pad)
        dr.text((12, y0 + tile_h // 3), f"{c}: {cname}", fill=(215, 222, 235))
        c_imgs = imgs[ys == c]
        for j in range(min(n_samples, len(c_imgs))):
            u8 = np.clip(c_imgs[j] * 255.0, 0, 255).astype(np.uint8)
            im = Image.fromarray(u8).resize((tile_w, tile_h), Image.Resampling.NEAREST).convert("RGB")
            canvas.paste(im, (left_lbl + j * (tile_w + pad), y0))
        if show_mean and len(c_imgs) > 0:
            mean_img = c_imgs.mean(axis=0)
            m_u8 = (np.clip(mean_img / max(float(mean_img.max()), 1e-6), 0, 1) * 255.0).astype(np.uint8)
            rgb = np.stack([(m_u8 * 0.35).astype(np.uint8), m_u8, m_u8], axis=-1)
            im_m = Image.fromarray(rgb, mode="RGB").resize((tile_w, tile_h), Image.Resampling.NEAREST)
            canvas.paste(im_m, (left_lbl + n_samples * (tile_w + pad) + 8, y0))
    return canvas


def filter_grid(
    weights: Any,
    *,
    ncols: int = 8,
    tile: int = 48,
    pad: int = 6,
    title: str = "Learned Signed Weights (orange = positive, blue = negative)",
) -> Image.Image:
    """Render a grid of signed 2D filters or PCA eigen-images (orange positive, blue negative)."""
    arr = _to_np(weights)
    n = len(arr)
    nrows = (n + ncols - 1) // ncols
    canvas = Image.new("RGB", (ncols * (tile + pad) + 20, nrows * (tile + pad) + 46), (16, 18, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((12, 12), title, fill=(240, 244, 250))
    for k in range(n):
        r, col = divmod(k, ncols)
        w = arr[k]
        w = w / (float(np.max(np.abs(w))) + 1e-6)
        pos, neg = np.clip(w, 0, 1), np.clip(-w, 0, 1)
        rgb = np.stack(
            [
                (pos * 250 + neg * 40).astype(np.uint8),
                (pos * 155 + neg * 140).astype(np.uint8),
                (pos * 45 + neg * 250).astype(np.uint8),
            ],
            axis=-1,
        )
        t_im = Image.fromarray(rgb, mode="RGB").resize((tile, tile), Image.Resampling.NEAREST)
        canvas.paste(t_im, (10 + col * (tile + pad), 40 + r * (tile + pad)))
    return canvas


def feature_maps(
    originals: Sequence[Any],
    maps: Any,
    row_labels: Sequence[str],
    *,
    n_maps: int = 8,
    tile: int = 56,
    title: str = "What the CNN 'Sees': Original Image (left) -> Learned Feature Maps (right)",
) -> Image.Image:
    """Render original images alongside their first ``n_maps`` CNN activation maps."""
    origs = [_to_np(o) for o in originals]
    fmaps = _to_np(maps)
    n_rows = len(origs)
    W = 100 + (n_maps + 1) * (tile + 4) + 12
    H = 40 + n_rows * (tile + 8) + 8
    canvas = Image.new("RGB", (W, H), (16, 18, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((12, 10), title, fill=(240, 244, 250))
    for r in range(n_rows):
        y0 = 40 + r * (tile + 8)
        dr.text((12, y0 + tile // 3), str(row_labels[r]), fill=(220, 228, 240))
        u8 = np.clip(origs[r] * 255.0, 0, 255).astype(np.uint8)
        orig = Image.fromarray(u8).resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(orig, (96, y0))
        for k in range(min(n_maps, fmaps.shape[1])):
            fm = fmaps[r, k]
            fm_u8 = (np.clip(fm / (float(fm.max()) + 1e-6), 0, 1) * 255.0).astype(np.uint8)
            rgb = np.stack([(fm_u8 * 0.25).astype(np.uint8), fm_u8, (fm_u8 * 0.85).astype(np.uint8)], axis=-1)
            canvas.paste(
                Image.fromarray(rgb, mode="RGB").resize((tile, tile), Image.Resampling.NEAREST),
                (96 + (k + 1) * (tile + 4) + 6, y0),
            )
    return canvas
