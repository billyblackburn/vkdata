"""Perceptually uniform & categorical 256-entry colormaps (colorcet-compatible) for Vulkan GPU."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


def _interp_stops(stops: Sequence[Tuple[float, Tuple[int, int, int]]], n: int = 256) -> np.ndarray:
    """Interpolate (pos, (r, g, b)) stops into an (n, 3) uint8 array."""
    out = np.zeros((n, 3), dtype=np.uint8)
    ts = np.linspace(0.0, 1.0, n)
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    rs = np.array([s[1][0] for s in stops], dtype=np.float64)
    gs = np.array([s[1][1] for s in stops], dtype=np.float64)
    bs = np.array([s[1][2] for s in stops], dtype=np.float64)
    out[:, 0] = np.clip(np.round(np.interp(ts, xs, rs)), 0, 255).astype(np.uint8)
    out[:, 1] = np.clip(np.round(np.interp(ts, xs, gs)), 0, 255).astype(np.uint8)
    out[:, 2] = np.clip(np.round(np.interp(ts, xs, bs)), 0, 255).astype(np.uint8)
    return out


def _hsv_rgb(n: int = 256, reverse: bool = False) -> np.ndarray:
    out = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        h = (1.0 - i / float(n)) if reverse else (i / float(n))
        h6 = (h % 1.0) * 6.0
        k = int(h6) % 6
        f = h6 - int(h6)
        if k == 0:   rgb = (1.0, f, 0.0)
        elif k == 1: rgb = (1.0 - f, 1.0, 0.0)
        elif k == 2: rgb = (0.0, 1.0, f)
        elif k == 3: rgb = (0.0, 1.0 - f, 1.0)
        elif k == 4: rgb = (f, 0.0, 1.0)
        else:        rgb = (1.0, 0.0, 1.0 - f)
        out[i] = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    return out


def _rgb_to_hex(arr: np.ndarray) -> List[str]:
    return [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in arr]


# Curated perceptually uniform colorcet + matplotlib + categorical palettes
_RGB_PALETTES: Dict[str, np.ndarray] = {
    "fire": _interp_stops([
        (0.00, (0, 0, 0)), (0.18, (60, 0, 20)), (0.36, (145, 10, 10)),
        (0.55, (220, 65, 0)), (0.74, (255, 155, 0)), (0.89, (255, 230, 80)), (1.00, (255, 255, 255)),
    ]),
    "bmy": _interp_stops([
        (0.00, (10, 8, 95)), (0.25, (75, 20, 155)), (0.50, (165, 45, 140)),
        (0.75, (235, 120, 75)), (1.00, (254, 235, 101)),
    ]),
    "bgy": _interp_stops([
        (0.00, (18, 28, 115)), (0.28, (25, 105, 145)), (0.58, (40, 175, 110)),
        (0.82, (145, 215, 65)), (1.00, (245, 245, 95)),
    ]),
    "bgyw": _interp_stops([
        (0.00, (16, 30, 110)), (0.25, (25, 110, 150)), (0.50, (45, 180, 115)),
        (0.75, (185, 225, 95)), (1.00, (252, 252, 245)),
    ]),
    "kbc": _interp_stops([
        (0.00, (0, 0, 0)), (0.25, (15, 40, 120)), (0.55, (30, 125, 205)),
        (0.82, (95, 215, 245)), (1.00, (220, 252, 255)),
    ]),
    "kb": _interp_stops([
        (0.00, (0, 0, 0)), (0.35, (15, 45, 135)), (0.70, (45, 115, 225)), (1.00, (165, 215, 255)),
    ]),
    "kg": _interp_stops([
        (0.00, (0, 0, 0)), (0.35, (12, 85, 35)), (0.70, (45, 185, 75)), (1.00, (185, 255, 185)),
    ]),
    "kr": _interp_stops([
        (0.00, (0, 0, 0)), (0.35, (125, 15, 15)), (0.70, (235, 60, 35)), (1.00, (255, 195, 165)),
    ]),
    "bmw": _interp_stops([
        (0.00, (15, 10, 115)), (0.35, (120, 35, 195)), (0.70, (220, 110, 225)), (1.00, (255, 245, 255)),
    ]),
    "viridis": _interp_stops([
        (0.00, (68, 1, 84)), (0.25, (59, 82, 139)), (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)), (1.00, (253, 231, 37)),
    ]),
    "inferno": _interp_stops([
        (0.00, (0, 0, 4)), (0.25, (87, 16, 110)), (0.50, (188, 55, 84)),
        (0.75, (249, 142, 9)), (1.00, (252, 255, 164)),
    ]),
    "plasma": _interp_stops([
        (0.00, (13, 8, 135)), (0.25, (126, 3, 168)), (0.50, (204, 71, 120)),
        (0.75, (248, 149, 64)), (1.00, (240, 249, 33)),
    ]),
    "magma": _interp_stops([
        (0.00, (0, 0, 4)), (0.25, (81, 18, 124)), (0.50, (183, 55, 121)),
        (0.75, (252, 137, 97)), (1.00, (252, 253, 191)),
    ]),
    "turbo": _interp_stops([
        (0.00, (48, 18, 59)), (0.20, (68, 119, 238)), (0.40, (27, 208, 213)),
        (0.60, (164, 252, 60)), (0.80, (251, 126, 33)), (1.00, (122, 4, 3)),
    ]),
    "coolwarm": _interp_stops([
        (0.00, (59, 76, 192)), (0.25, (125, 160, 245)), (0.50, (221, 221, 221)),
        (0.75, (244, 152, 122)), (1.00, (180, 4, 38)),
    ]),
    "bwr": _interp_stops([
        (0.00, (20, 60, 220)), (0.50, (240, 240, 245)), (1.00, (220, 35, 35)),
    ]),
    "bkr": _interp_stops([
        (0.00, (50, 145, 255)), (0.50, (18, 20, 26)), (1.00, (255, 70, 50)),
    ]),
    "rainbow": _interp_stops([
        (0.00, (55, 35, 180)), (0.25, (25, 160, 220)), (0.50, (60, 205, 95)),
        (0.75, (245, 195, 35)), (1.00, (225, 45, 35)),
    ]),
    "gray": _interp_stops([(0.00, (0, 0, 0)), (1.00, (255, 255, 255))]),
    "blues": _interp_stops([(0.00, (8, 24, 64)), (0.50, (45, 130, 215)), (1.00, (235, 248, 255))]),
    "greens": _interp_stops([(0.00, (5, 40, 20)), (0.50, (40, 170, 85)), (1.00, (235, 255, 235))]),
    "reds": _interp_stops([(0.00, (55, 5, 5)), (0.50, (215, 45, 35)), (1.00, (255, 235, 225))]),
    "hsv": _hsv_rgb(256, reverse=False),
    "hsv_r": _hsv_rgb(256, reverse=True),
}

# Add reversed `_r` versions for all sequential/diverging colormaps
for _k in list(_RGB_PALETTES.keys()):
    if not _k.endswith("_r") and f"{_k}_r" not in _RGB_PALETTES:
        _RGB_PALETTES[f"{_k}_r"] = _RGB_PALETTES[_k][::-1].copy()

# High-contrast categorical palettes (Glasbey / Tab10 style)
_CATEGORICAL_HEX: Dict[str, List[str]] = {
    "glasbey": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#e377c2", "#bcbd22",
        "#ff9896", "#98df8a", "#aec7e8", "#ffbb78", "#c5b0d5", "#f7b6d2", "#dbdb8d", "#9edae5",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354",
    ],
    "glasbey_dark": [
        "#ff6b6b", "#4ecdc4", "#ffe66d", "#5c9ead", "#ff9f1c", "#a29bfe", "#55efc4", "#fd79a8",
        "#74b9ff", "#fab1a0", "#81ecec", "#ffeaa7", "#e17055", "#00cec9", "#6c5ce7", "#fdcb6e",
    ],
    "category10": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ],
}

palette: Dict[str, List[str]] = {k: _rgb_to_hex(v) for k, v in _RGB_PALETTES.items()}
palette.update(_CATEGORICAL_HEX)

_LUT_GPU_CACHE: Dict[str, Any] = {}


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    s = h.strip().lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def get_rgb_lut(cmap: Any = "fire") -> np.ndarray:
    """Return a (256, 3) uint8 RGB LUT for any colormap name or list of hex colors."""
    if isinstance(cmap, str):
        key = cmap.strip().lower()
        if key in _RGB_PALETTES:
            return _RGB_PALETTES[key]
        if key in _CATEGORICAL_HEX:
            cols = _CATEGORICAL_HEX[key]
            arr = np.array([_hex_to_rgb(cols[i % len(cols)]) for i in range(256)], dtype=np.uint8)
            return arr
        raise KeyError(f"Unknown colormap {cmap!r}; available: {sorted(palette.keys())[:16]}...")
    if isinstance(cmap, (list, tuple)):
        if len(cmap) == 256 and isinstance(cmap[0], str):
            return np.array([_hex_to_rgb(c) for c in cmap], dtype=np.uint8)
        stops = [(i / max(1, len(cmap) - 1), _hex_to_rgb(c) if isinstance(c, str) else tuple(c[:3])) for i, c in enumerate(cmap)]
        return _interp_stops(stops, 256)
    arr = np.asarray(cmap)
    if arr.shape == (256, 3):
        return arr.astype(np.uint8)
    raise ValueError(f"Unsupported cmap specification: {type(cmap).__name__}")


def get_lut_u32(cmap: Any = "fire") -> np.ndarray:
    """Return a 256-entry packed RGBA uint32 LUT for Vulkan GPU shaders."""
    rgb = get_rgb_lut(cmap).astype(np.uint32)
    return rgb[:, 0] | (rgb[:, 1] << 8) | (rgb[:, 2] << 16) | (np.uint32(0xFF) << 24)


def get_lut_gpu(cmap: Any = "fire") -> Any:
    """Return a cached 256-entry int32 vknumpy GPU buffer holding packed RGBA8 LUT entries."""
    import vknumpy as vnp

    if isinstance(cmap, str):
        key = cmap.strip().lower()
        if key in _LUT_GPU_CACHE:
            return _LUT_GPU_CACHE[key]
        u32 = get_lut_u32(key)
        g = vnp.array(u32.view(np.int32), dtype=np.int32)
        _LUT_GPU_CACHE[key] = g
        return g
    u32 = get_lut_u32(cmap)
    return vnp.array(u32.view(np.int32), dtype=np.int32)


def swatches(names: Sequence[str] | None = None) -> Any:
    """Display an HTML visual swatch table of vkcolorcet colormaps in Jupyter."""
    from IPython.display import HTML

    keys = list(names) if names is not None else [
        "fire", "bmy", "bgy", "bgyw", "kbc", "kb", "kg", "kr", "bmw",
        "viridis", "inferno", "plasma", "magma", "turbo",
        "coolwarm", "bwr", "bkr", "rainbow", "gray", "glasbey", "glasbey_dark",
    ]
    rows = []
    for k in keys:
        cols = palette[k]
        if k in _CATEGORICAL_HEX:
            stops = "".join(f"<span style='display:inline-block;width:14px;height:18px;background:{c}'></span>" for c in cols[:16])
            bar = f"<div style='display:flex;border-radius:3px;overflow:hidden'>{stops}</div>"
        else:
            grad = ", ".join(cols[i] for i in range(0, 256, 16))
            bar = f"<div style='height:18px;width:320px;border-radius:3px;background:linear-gradient(to right, {grad})'></div>"
        rows.append(
            f"<tr><td style='padding:3px 10px;font-family:monospace;color:#e5e9f0'>{k}</td>"
            f"<td style='padding:3px 8px'>{bar}</td></tr>"
        )
    return HTML(
        "<div style='background:#181b24;padding:10px;border:1px solid #3a4052;border-radius:6px;display:inline-block'>"
        "<div style='color:#ff9128;font-weight:bold;font-family:sans-serif;margin-bottom:6px'>vkcolorcet Colormaps</div>"
        f"<table>{''.join(rows)}</table></div>"
    )
