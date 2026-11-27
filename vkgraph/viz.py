"""vkgraph.viz: SynthWave '84 spring-layout graph visualization (zero matplotlib dependency)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageDraw

from .graph import Graph

_PALETTE = ["#36f9f6", "#ff7edb", "#fede5d", "#72f1b8", "#ff9e64", "#c792ea", "#60a5fa"]


def spring_layout(graph: Graph, *, iterations: int = 80, seed: int = 42) -> np.ndarray:
    """Compute a clean 2D Fruchterman-Reingold force-directed layout for small/medium graphs."""
    n = graph.num_nodes
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    k = float(np.sqrt(1.0 / max(n, 1)))
    t = 0.25

    src, dst = graph.src, graph.dst
    for _ in range(int(iterations)):
        # Pairwise repulsive forces
        diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-4)  # (N, N)
        rep = (k * k / dist) * (1.0 / dist)
        np.fill_diagonal(rep, 0.0)
        disp = np.sum(diff * rep[:, :, None], axis=1)

        # Attractive forces along edges
        if len(src) > 0:
            ediff = pos[src] - pos[dst]
            edist = np.sqrt(np.sum(ediff * ediff, axis=1) + 1e-4)
            att = (edist / k)[:, None] * ediff
            np.add.at(disp, src, -att)
            np.add.at(disp, dst, att)

        dnorm = np.sqrt(np.sum(disp * disp, axis=1, keepdims=True) + 1e-6)
        pos += (disp / dnorm) * np.minimum(dnorm, t)
        t *= 0.96

    pos -= pos.mean(axis=0, keepdims=True)
    scale = np.maximum(np.abs(pos).max(axis=0, keepdims=True), 1e-5)
    return (pos / scale).astype(np.float32)


def render_graph(
    graph: Graph,
    *,
    labels: Optional[Any] = None,
    node_sizes: Optional[Any] = None,
    highlight_path: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
    show_ids: bool = True,
    title: str = "",
    width: int = 860,
    height: int = 520,
    seed: int = 42,
) -> Image.Image:
    """Render a Graph with glowing SynthWave '84 edges, sized nodes, optional highlighted path & legend."""
    pos = spring_layout(graph, seed=seed)
    pal = list(palette) if palette is not None else list(_PALETTE)

    pad_x, pad_t, pad_b = 54, 48, 36
    pw, ph = width - 2 * pad_x, height - pad_t - pad_b
    px = pad_x + ((pos[:, 0] + 1.0) * 0.5) * pw
    py = pad_t + ((1.0 - pos[:, 1]) * 0.5) * ph

    img = Image.new("RGB", (width, height), "#1b1728")
    dr = ImageDraw.Draw(img)
    if title:
        dr.text((16, 12), title, fill="#f4eee4")

    # Draw base edges
    for u, v in zip(graph.src.tolist(), graph.dst.tolist()):
        dr.line([(float(px[u]), float(py[u])), (float(px[v]), float(py[v]))], fill="#3b355c", width=1)

    # Highlight shortest path if provided
    path_set = set(highlight_path) if highlight_path else set()
    if highlight_path and len(highlight_path) > 1:
        for a, b in zip(highlight_path[:-1], highlight_path[1:]):
            dr.line([(float(px[a]), float(py[a])), (float(px[b]), float(py[b]))], fill="#fede5d", width=4)

    # Node colors & radii
    lab_np = (
        (labels.to_numpy() if hasattr(labels, "to_numpy") else np.asarray(labels)).astype(np.int32).ravel()
        if labels is not None
        else np.zeros(graph.num_nodes, dtype=np.int32)
    )
    if node_sizes is not None:
        sz = (node_sizes.to_numpy() if hasattr(node_sizes, "to_numpy") else np.asarray(node_sizes)).astype(np.float32).ravel()
        lo, hi = float(sz.min()), float(sz.max())
        radii = 8.0 + 14.0 * (sz - lo) / max(hi - lo, 1e-6)
    else:
        radii = np.full(graph.num_nodes, 11.0, dtype=np.float32)

    for i in range(graph.num_nodes):
        r = float(radii[i])
        cx, cy = float(px[i]), float(py[i])
        col = pal[int(lab_np[i]) % len(pal)]
        outline = "#fede5d" if i in path_set else "#1b1728"
        ow = 3 if i in path_set else 2
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=outline, width=ow)
        if show_ids and graph.num_nodes <= 60:
            dr.text((cx - (6 if i >= 10 else 3), cy - 5), str(i), fill="#1b1728")

    if class_names:
        box_w = max(135, max(len(str(c)) for c in class_names) * 7 + 40)
        box_h = len(class_names) * 24 + 16
        lx, ly = width - box_w - 16, height - box_h - 16
        dr.rounded_rectangle([lx, ly, lx + box_w, ly + box_h], radius=8, fill="#262335", outline="#c792ea", width=2)
        for i, cname in enumerate(class_names):
            yy = ly + 10 + i * 24
            dr.ellipse([lx + 10, yy + 2, lx + 22, yy + 14], fill=pal[i % len(pal)])
            dr.text((lx + 30, yy + 1), str(cname), fill="#f4eee4")
    return img
