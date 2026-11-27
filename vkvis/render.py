"""Vulkan GPU embedding scatter, 3D sphere, comparison-grid, and animation renderers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as _np
from PIL import Image, ImageDraw
import vkcolorcet as cc
import vkpandas as vpd
import vkshader as vks


_DEFAULT_CAT_PALETTE = [
    "#38bdf8", "#f97316", "#22c55e", "#e879f9", "#facc15",
    "#f43f5e", "#2dd4bf", "#a78bfa", "#fb7185", "#84cc16",
    "#60a5fa", "#fbbf24", "#34d399", "#c084fc", "#f87171",
    "#06b6d4",
]


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _bounds(arr: _np.ndarray, pad: float = 0.08):
    lo = float(_np.percentile(arr, 0.5))
    hi = float(_np.percentile(arr, 99.5))
    span = max(hi - lo, 1e-4)
    return (lo - span * pad, hi + span * pad)


def _splat_2d_circles(
    xs: _np.ndarray,
    ys: _np.ndarray,
    rgb_pts: _np.ndarray,
    xr: Tuple[float, float],
    yr: Tuple[float, float],
    width: int,
    height: int,
    radius: int,
    bg_rgb: Tuple[int, int, int],
) -> Image.Image:
    """Fast vectorized 2D circle-splatting scatter renderer (zero matplotlib dependency)."""
    canvas = _np.empty((height, width, 3), dtype=_np.uint8)
    canvas[:, :, 0] = bg_rgb[0]
    canvas[:, :, 1] = bg_rgb[1]
    canvas[:, :, 2] = bg_rgb[2]

    px = _np.round((xs - xr[0]) / max(xr[1] - xr[0], 1e-6) * (width - 1)).astype(_np.int32)
    py = _np.round((1.0 - (ys - yr[0]) / max(yr[1] - yr[0], 1e-6)) * (height - 1)).astype(_np.int32)

    r = max(1, int(radius))
    # Precompute circular disk offsets
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    for dy, dx in offs:
        xx = px + dx
        yy = py + dy
        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        canvas[yy[valid], xx[valid]] = rgb_pts[valid]

    return Image.fromarray(canvas, mode="RGB")


def render_embedding(
    Y,
    labels=None,
    *,
    width: int = 640,
    height: int = 480,
    palette: Optional[Sequence[str]] = None,
    cmap: str = "turbo",
    background: str = "#0a0d14",
    point_size: int = 2,
    title: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    legend_loc: str = "lower right",
    clip_percentile: Optional[float] = None,
    yaw: float = 0.55,
    pitch: float = 0.40,
    dist: float = 2.1,
) -> Image.Image:
    """Render a 2D or 3D embedding using ``vkshader`` + ``vkcolorcet`` with optional legend."""
    if hasattr(Y, "_columns") and hasattr(Y, "columns"):
        Y_np = _np.column_stack([Y[c].to_numpy() for c in Y.columns]).astype(_np.float32)
    elif hasattr(Y, "to_numpy"):
        Y_np = _np.asarray(Y.to_numpy(), dtype=_np.float32)
    elif hasattr(Y, "numpy") and callable(Y.numpy):
        Y_np = _np.asarray(Y.numpy(), dtype=_np.float32)
    else:
        Y_np = _np.asarray(Y, dtype=_np.float32)
    if Y_np.ndim != 2 or Y_np.shape[1] not in (2, 3):
        raise ValueError(f"Expected 2D or 3D embedding (N, 2|3), got {Y_np.shape}")
    if clip_percentile is not None and 0.0 < float(clip_percentile) < 50.0:
        p = float(clip_percentile)
        lo, hi = _np.percentile(Y_np, [p, 100.0 - p], axis=0)
        Y_np = _np.clip(Y_np, lo, hi)

    n, d = Y_np.shape
    xr = _bounds(Y_np[:, 0])
    yr = _bounds(Y_np[:, 1])
    bg_rgb = _hex_to_rgb(background) if background.startswith("#") else (10, 13, 20)

    is_cat = False
    codes = None
    vals = None
    if labels is not None:
        if hasattr(labels, "to_numpy"):
            lab_np = _np.asarray(labels.to_numpy())
        elif hasattr(labels, "numpy") and callable(labels.numpy):
            lab_np = _np.asarray(labels.numpy())
        else:
            lab_np = _np.asarray(labels)
        is_cat = _np.issubdtype(lab_np.dtype, _np.integer) or lab_np.dtype.kind in ("U", "S", "O")
        if is_cat:
            _, codes = _np.unique(lab_np, return_inverse=True)
            codes = codes.astype(_np.int32)
            pal = list(palette) if palette is not None else list(_DEFAULT_CAT_PALETTE)
            n_c = int(codes.max()) + 1 if codes.size else 1
            while len(pal) < n_c:
                pal.extend(cc.glasbey_dark)
        else:
            vals = lab_np.astype(_np.float32)
            pal = list(palette) if palette is not None else list(getattr(cc, cmap, cc.turbo))
    else:
        pal = list(palette) if palette is not None else list(getattr(cc, cmap, cc.fire))

    if d == 3:
        zr = _bounds(Y_np[:, 2])
        if labels is not None and is_cat:
            df = vpd.DataFrame({"x": Y_np[:, 0], "y": Y_np[:, 1], "z": Y_np[:, 2], "c": codes})
            shaded = vks.render_3d(
                df, "x", "y", "z", width=width, height=height,
                x_range=xr, y_range=yr, z_range=zr,
                kind="spheres" if n <= 25000 else "points",
                color_by="c", color_key=pal, basemap=None,
                yaw=yaw, pitch=pitch, dist=dist, z_scale=0.85,
            )
        elif labels is not None:
            v_lo, v_hi = float(_np.min(vals)), float(_np.max(vals))
            bins = _np.clip(((vals - v_lo) / max(v_hi - v_lo, 1e-6) * 31.0).astype(_np.int32), 0, 31)
            pal32 = [pal[int(i * (len(pal) - 1) / 31)] for i in range(32)]
            df = vpd.DataFrame({"x": Y_np[:, 0], "y": Y_np[:, 1], "z": Y_np[:, 2], "c": bins})
            shaded = vks.render_3d(
                df, "x", "y", "z", width=width, height=height,
                x_range=xr, y_range=yr, z_range=zr,
                kind="spheres" if n <= 25000 else "points",
                color_by="c", color_key=pal32, basemap=None,
                yaw=yaw, pitch=pitch, dist=dist, z_scale=0.85,
            )
        else:
            df = vpd.DataFrame({"x": Y_np[:, 0], "y": Y_np[:, 1], "z": Y_np[:, 2]})
            shaded = vks.render_3d(
                df, "x", "y", "z", width=width, height=height,
                x_range=xr, y_range=yr, z_range=zr,
                kind="points", cmap=cmap, basemap=None,
                yaw=yaw, pitch=pitch, dist=dist, z_scale=0.85,
            )
        im = Image.fromarray(shaded.rgba[:, :, :3], mode="RGB")
    elif labels is None:
        # Pure GPU density shading via vkshader.Canvas
        cvs = vks.Canvas(plot_width=width, plot_height=height, x_range=xr, y_range=yr)
        df = vpd.DataFrame({"x": Y_np[:, 0], "y": Y_np[:, 1]})
        agg = cvs.points(df, "x", "y")
        shaded = vks.shade(agg, cmap=cmap)
        rgba = shaded.rgba
        alpha = rgba[:, :, 3:4].astype(_np.float32) / 255.0
        bg_arr = _np.array(bg_rgb, dtype=_np.float32).reshape(1, 1, 3)
        comp = _np.clip(rgba[:, :, :3].astype(_np.float32) * alpha + bg_arr * (1.0 - alpha), 0, 255).astype(_np.uint8)
        im = Image.fromarray(comp, mode="RGB")
    else:
        # Crisp 2D circle-splatting scatter plot colored by vkcolorcet palette
        pal_rgb = _np.array([_hex_to_rgb(c) for c in pal], dtype=_np.uint8)
        if is_cat:
            rgb_pts = pal_rgb[codes % len(pal_rgb)]
        else:
            v_lo, v_hi = float(_np.min(vals)), float(_np.max(vals))
            norm = _np.clip((vals - v_lo) / max(v_hi - v_lo, 1e-6), 0.0, 1.0)
            idx = _np.round(norm * (len(pal_rgb) - 1)).astype(_np.int32)
            rgb_pts = pal_rgb[idx]
        im = _splat_2d_circles(Y_np[:, 0], Y_np[:, 1], rgb_pts, xr, yr, width, height, point_size, bg_rgb)

    if title:
        bar_h = 24
        out = Image.new("RGB", (width, height + bar_h), (12, 16, 26))
        draw = ImageDraw.Draw(out)
        draw.text((10, 5), str(title), fill=(235, 240, 248))
        out.paste(im, (0, bar_h))
        im = out

    if class_names:
        draw = ImageDraw.Draw(im)
        box_w = max(130, max(len(str(c)) for c in class_names) * 7 + 40)
        box_h = len(class_names) * 24 + 16
        lx = 16 if "left" in legend_loc else im.width - box_w - 16
        ly = 36 if "upper" in legend_loc else im.height - box_h - 16
        draw.rounded_rectangle(
            [lx, ly, lx + box_w, ly + box_h],
            radius=8, fill="#262335", outline="#ff7edb", width=2,
        )
        for i, cname in enumerate(class_names):
            yy = ly + 10 + i * 24
            col = pal[i % len(pal)]
            draw.ellipse([lx + 10, yy + 2, lx + 22, yy + 14], fill=col)
            draw.text((lx + 30, yy + 1), str(cname), fill="#f4eee4")
    return im


def compare_embeddings(
    embeddings: Dict[str, _np.ndarray],
    labels=None,
    *,
    width: int = 380,
    height: int = 320,
    ncols: int = 3,
    point_size: int = 2,
    palette: Optional[Sequence[str]] = None,
    cmap: str = "turbo",
    background: str = "#0a0d14",
) -> Image.Image:
    """Render a side-by-side comparison grid of multiple embeddings."""
    items = list(embeddings.items())
    if not items:
        raise ValueError("embeddings dict is empty")
    nc = min(int(ncols), len(items))
    nr = (len(items) + nc - 1) // nc
    pad = 8
    bar_h = 24
    tile_h = height + bar_h
    grid = Image.new("RGB", (nc * (width + pad) + pad, nr * (tile_h + pad) + pad), (8, 10, 16))

    for idx, (name, Y) in enumerate(items):
        r, c = divmod(idx, nc)
        im = render_embedding(
            Y, labels=labels, width=width, height=height,
            point_size=point_size, palette=palette, cmap=cmap,
            background=background, title=name,
        )
        grid.paste(im, (pad + c * (width + pad), pad + r * (tile_h + pad)))
    return grid


def animate_embedding(
    history: Sequence[_np.ndarray],
    labels=None,
    *,
    width: int = 560,
    height: int = 420,
    point_size: int = 2,
    fps: int = 12,
    title_prefix: str = "Step",
    save_path: Optional[Union[str, Path]] = None,
) -> List[Image.Image]:
    """Render an optimization trajectory into PIL frames (and optionally save a GIF)."""
    frames: List[Image.Image] = []
    total = len(history)
    for i, Y in enumerate(history):
        im = render_embedding(
            Y, labels=labels, width=width, height=height,
            point_size=point_size, title=f"{title_prefix} ({i + 1}/{total})",
        )
        frames.append(im)
    if save_path is not None and frames:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            str(p),
            save_all=True,
            append_images=frames[1:],
            duration=max(20, int(1000 / max(1, fps))),
            loop=0,
        )
    return frames
