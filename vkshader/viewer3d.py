"""Zero-copy Vulkan GPU 3D perspective visualization & interactive orbit viewer."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import anywidget
import numpy as np
import traitlets
import vknumpy as vnp

from ._backend import render_3d_gpu
from .basemap import get_esri_mosaic_gpu
from .canvas import ShadedImage, _CAT_CODES_CACHE, _build_palette_gpu, _extract_gpu_col


def build_camera_params(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    width: int,
    height: int,
    yaw: float = 0.55,
    pitch: float = 0.52,
    dist: float = 2.55,
    z_scale: float = 0.55,
    fov_deg: float = 48.0,
) -> np.ndarray:
    """Pack 24 float32 camera & world-normalization parameters for raster_3d.comp."""
    cx = 0.5 * (x_range[0] + x_range[1])
    cy = 0.5 * (y_range[0] + y_range[1])
    half_span = max(0.5 * max(x_range[1] - x_range[0], y_range[1] - y_range[0]), 1e-6)
    zmin, zmax = float(z_range[0]), float(z_range[1])
    inv_dz = 1.0 / max(zmax - zmin, 1e-6)

    p = max(0.08, min(1.48, float(pitch)))
    y = float(yaw)
    d = max(0.5, min(10.0, float(dist)))

    target = np.array([0.0, 0.0, z_scale * 0.22], dtype=np.float64)
    cam_pos = target + d * np.array([
        math.sin(y) * math.cos(p),
        -math.cos(y) * math.cos(p),
        math.sin(p),
    ], dtype=np.float64)

    fwd = target - cam_pos
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= rn
    up = np.cross(right, fwd)

    focal = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    aspect = float(width) / max(float(height), 1.0)

    params = np.zeros(24, dtype=np.float32)
    params[0:3] = cam_pos
    params[3:6] = right
    params[6:9] = up
    params[9:12] = fwd
    params[12] = cx
    params[13] = cy
    params[14] = 1.0 / half_span
    params[15] = zmin
    params[16] = inv_dz
    params[17] = float(z_scale)
    params[18] = focal
    params[19] = aspect
    params[20] = float(x_range[0])
    params[21] = float(x_range[1])
    params[22] = float(y_range[0])
    params[23] = float(y_range[1])
    return params


def _resolve_cat_codes(source: Any, column: str) -> Tuple[vnp.ndarray, int]:
    col_obj = source[column]
    cache_key = (id(source), str(column))
    cached = _CAT_CODES_CACHE.get(cache_key)
    if cached is not None and cached[0] is col_obj:
        return cached[1], cached[2]

    if hasattr(col_obj, "values") and isinstance(col_obj.values, vnp.ndarray):
        codes = col_obj.values if col_obj.values.dtype == np.int32 else col_obj.values.astype(np.int32)
        n_cats = max(2, int(vnp.max(codes).item()) + 1)
    else:
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            raw = col_obj.values if hasattr(col_obj, "values") else col_obj
            enc = pc.dictionary_encode(pa.array(raw))
            codes = vnp.array(enc.indices.to_numpy(zero_copy_only=False).astype(np.int32), dtype=np.int32)
            n_cats = max(2, len(enc.dictionary))
        except Exception:
            arr = np.asarray(col_obj)
            if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
                codes = vnp.array(arr.astype(np.int32), dtype=np.int32)
                n_cats = max(2, int(arr.max()) + 1) if arr.size else 2
            else:
                _, inv = np.unique(arr, return_inverse=True)
                codes = vnp.array(inv.astype(np.int32), dtype=np.int32)
                n_cats = max(2, int(inv.max()) + 1) if inv.size else 2
    if len(_CAT_CODES_CACHE) > 16:
        _CAT_CODES_CACHE.clear()
    _CAT_CODES_CACHE[cache_key] = (col_obj, codes, n_cats)
    return codes, n_cats


def render_3d(
    source: Any,
    x: str,
    y: str,
    z: str,
    width: int = 760,
    height: int = 540,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    kind: str = "lines",
    color_by: Optional[str] = None,
    color_key: Any = None,
    basemap: Optional[str] = "esri",
    yaw: float = 0.55,
    pitch: float = 0.52,
    dist: float = 2.55,
    z_scale: float = 0.55,
    alpha: int = 235,
) -> ShadedImage:
    """Render a 3D perspective volumetric view of `(x, y, z)` lines or points on Vulkan GPU."""
    xs = _extract_gpu_col(source, x)
    if xs.dtype != np.float32:
        xs = xs.astype(np.float32)
    ys = _extract_gpu_col(source, y)
    if ys.dtype != np.float32:
        ys = ys.astype(np.float32)
    zs = _extract_gpu_col(source, z)
    if zs.dtype != np.float32:
        zs = zs.astype(np.float32)

    xr = x_range if x_range is not None else (float(vnp.nanmin(xs).item()), float(vnp.nanmax(xs).item()))
    yr = y_range if y_range is not None else (float(vnp.nanmin(ys).item()), float(vnp.nanmax(ys).item()))
    zr = z_range if z_range is not None else (float(vnp.nanmin(zs).item()), float(vnp.nanmax(zs).item()))

    # Make the ground-plane square in Web Mercator meters so aspect is distortion-free
    cx = 0.5 * (xr[0] + xr[1])
    cy = 0.5 * (yr[0] + yr[1])
    hs = max(0.5 * max(xr[1] - xr[0], yr[1] - yr[0]), 1.0)
    sq_xr = (cx - hs, cx + hs)
    sq_yr = (cy - hs, cy + hs)

    cam = build_camera_params(sq_xr, sq_yr, zr, width, height, yaw=yaw, pitch=pitch, dist=dist, z_scale=z_scale)

    vals = None
    pal = None
    is_lines = (kind == "lines")
    if color_by is not None:
        vals, n_cats = _resolve_cat_codes(source, color_by)
        pal = _build_palette_gpu(color_key, n_cats=n_cats)
        mode = 2 if is_lines else 3
    else:
        mode = 1 if is_lines else 0

    bg_gpu = None
    uv_rect = (0.0, 0.0, 1.0, 1.0)
    if basemap == "esri":
        res = get_esri_mosaic_gpu(sq_xr, sq_yr, width, height)
        if res is not None:
            bg_gpu, uv_rect = res

    rgba_gpu = render_3d_gpu(
        xs, ys, zs, cam, width, height,
        mode=mode, vals=vals, palette=pal,
        alpha=float(alpha) / 255.0,
        bg=bg_gpu, uv_rect=uv_rect,
    )
    rgba = rgba_gpu.to_numpy().view(np.uint8).reshape(height, width, 4)
    return ShadedImage(rgba=rgba, x_range=sq_xr, y_range=sq_yr)


_ESM_3D = r"""
function render({ model, el }) {
  const w = model.get("width");
  const h = model.get("height");

  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.width = `${w}px`;
  wrap.style.height = `${h}px`;
  wrap.style.overflow = "hidden";
  wrap.style.background = "#0b0f18";
  wrap.style.border = "1px solid #3a4052";
  wrap.style.borderRadius = "6px";
  wrap.style.userSelect = "none";
  wrap.style.touchAction = "none";
  wrap.style.cursor = "grab";

  const img = document.createElement("img");
  img.draggable = false;
  img.style.width = "100%";
  img.style.height = "100%";
  img.style.display = "block";
  img.style.pointerEvents = "none";
  wrap.appendChild(img);

  const hint = document.createElement("div");
  hint.textContent = "Drag: 3D Orbit (Yaw/Pitch) | Wheel: Zoom | Dbl-click: Reset";
  hint.style.position = "absolute";
  hint.style.left = "8px";
  hint.style.bottom = "6px";
  hint.style.padding = "2px 7px";
  hint.style.fontSize = "11px";
  hint.style.fontFamily = "monospace";
  hint.style.color = "#e5e9f0";
  hint.style.background = "rgba(10, 14, 22, 0.78)";
  hint.style.borderRadius = "3px";
  hint.style.pointerEvents = "none";
  wrap.appendChild(hint);

  let prevUrl = null;
  function updateImage() {
    const data = model.get("image_bytes");
    if (!data || data.byteLength === 0) return;
    const blob = new Blob([data], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    img.src = url;
    if (prevUrl) URL.revokeObjectURL(prevUrl);
    prevUrl = url;
  }
  model.on("change:image_bytes", updateImage);
  updateImage();

  let lastSend = 0;
  function sendCamera(yaw, pitch, dist, force = false) {
    const now = performance.now();
    if (!force && now - lastSend < 45) return;
    lastSend = now;
    model.set("camera", [yaw, pitch, dist]);
    model.save_changes();
  }

  let dragging = false;
  let sx = 0, sy = 0;
  let startCam = null;

  wrap.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    wrap.setPointerCapture(e.pointerId);
    sx = e.clientX; sy = e.clientY;
    startCam = [...model.get("camera")];
    dragging = true;
    wrap.style.cursor = "grabbing";
  });

  wrap.addEventListener("pointermove", (e) => {
    if (!dragging || !startCam) return;
    const dx = e.clientX - sx;
    const dy = e.clientY - sy;
    const nyaw = startCam[0] + dx * 0.008;
    const npitch = Math.max(0.10, Math.min(1.46, startCam[1] + dy * 0.007));
    sendCamera(nyaw, npitch, startCam[2], false);
  });

  wrap.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    dragging = false;
    wrap.style.cursor = "grab";
    if (startCam) {
      const dx = e.clientX - sx;
      const dy = e.clientY - sy;
      const nyaw = startCam[0] + dx * 0.008;
      const npitch = Math.max(0.10, Math.min(1.46, startCam[1] + dy * 0.007));
      sendCamera(nyaw, npitch, startCam[2], true);
    }
    startCam = null;
  });

  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [yaw, pitch, dist] = model.get("camera");
    const factor = e.deltaY > 0 ? 1.12 : 0.89;
    const ndist = Math.max(0.7, Math.min(7.5, dist * factor));
    sendCamera(yaw, pitch, ndist, true);
  }, { passive: false });

  wrap.addEventListener("dblclick", () => {
    sendCamera(0.55, 0.52, 2.45, true);
  });

  el.appendChild(wrap);
}
export default { render };
"""


class Interactive3DWidget(anywidget.AnyWidget):
    """Drag-to-orbit & wheel-to-zoom 3D Vulkan GPU viewport widget."""

    _esm = _ESM_3D
    width = traitlets.Int(760).tag(sync=True)
    height = traitlets.Int(520).tag(sync=True)
    image_bytes = traitlets.Bytes(b"").tag(sync=True)
    camera = traitlets.List(traitlets.Float(), default_value=[0.55, 0.52, 2.45]).tag(sync=True)

    def __init__(
        self,
        width: int,
        height: int,
        camera: Tuple[float, float, float],
        image_bytes: bytes,
        on_camera_change: Callable[[float, float, float], None],
    ) -> None:
        super().__init__(
            width=int(width),
            height=int(height),
            camera=[float(camera[0]), float(camera[1]), float(camera[2])],
            image_bytes=image_bytes,
        )
        self._on_camera_change = on_camera_change
        self.observe(self._handle_cam, names=["camera"])

    def _handle_cam(self, change: dict) -> None:
        cam = change["new"]
        if len(cam) == 3:
            self._on_camera_change(float(cam[0]), float(cam[1]), float(cam[2]))


def interact_3d(
    source: Any,
    x: str,
    y: str,
    z: str,
    width: int = 760,
    height: int = 520,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    kind: str = "lines",
    color_modes: Optional[Dict[str, Tuple[Optional[str], Any]]] = None,
    regions: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    basemap: Optional[str] = "esri",
    z_scale: float = 0.55,
    jpeg_quality: int = 86,
) -> Any:
    """Launch an interactive 3D Vulkan GPU orbit viewer inside Jupyter!"""
    import ipywidgets as widgets
    from IPython.display import display

    modes = color_modes or {"Height / Z (Fire)": (None, None)}
    mode_names = list(modes.keys())
    init_xr = x_range
    init_yr = y_range
    if regions:
        first_reg = next(iter(regions.keys()))
        init_xr, init_yr = regions[first_reg]

    state: Dict[str, Any] = {
        "yaw": 0.55,
        "pitch": 0.52,
        "dist": 2.45,
        "z_scale": float(z_scale),
        "xr": init_xr,
        "yr": init_yr,
        "mode": mode_names[0],
        "basemap": basemap,
    }

    def _draw() -> Tuple[bytes, float]:
        t0 = time.perf_counter()
        col_by, col_key = modes[state["mode"]]
        img = render_3d(
            source, x, y, z,
            width=width, height=height,
            x_range=state["xr"], y_range=state["yr"], z_range=z_range,
            kind=kind, color_by=col_by, color_key=col_key,
            basemap=state["basemap"],
            yaw=state["yaw"], pitch=state["pitch"], dist=state["dist"],
            z_scale=state["z_scale"],
        )
        jpg = img.to_jpeg(quality=jpeg_quality)
        return jpg, (time.perf_counter() - t0) * 1000.0

    jpg0, ms0 = _draw()
    status = widgets.Label(
        value=f"3D Vulkan GPU | {ms0:.1f} ms | yaw={math.degrees(state['yaw']):.0f}° pitch={math.degrees(state['pitch']):.0f}°"
    )

    view_w = None

    def _refresh() -> None:
        jpg, ms = _draw()
        assert view_w is not None
        view_w.image_bytes = jpg
        status.value = (
            f"3D Vulkan GPU | {ms:.1f} ms | yaw={math.degrees(state['yaw']):.0f}° "
            f"pitch={math.degrees(state['pitch']):.0f}° dist={state['dist']:.2f}"
        )

    def _on_cam(yaw: float, pitch: float, dist: float) -> None:
        state["yaw"], state["pitch"], state["dist"] = yaw, pitch, dist
        _refresh()

    view_w = Interactive3DWidget(
        width=width, height=height,
        camera=(state["yaw"], state["pitch"], state["dist"]),
        image_bytes=jpg0,
        on_camera_change=_on_cam,
    )

    controls = []
    if regions:
        reg_dd = widgets.Dropdown(options=list(regions.keys()), value=next(iter(regions.keys())), description="Region:")
        reg_dd.observe(lambda ch: (state.update({"xr": regions[ch["new"]][0], "yr": regions[ch["new"]][1]}), _refresh()), names="value")
        controls.append(reg_dd)

    if len(mode_names) > 1:
        mode_dd = widgets.Dropdown(options=mode_names, value=mode_names[0], description="Color:")
        mode_dd.observe(lambda ch: (state.update({"mode": ch["new"]}), _refresh()), names="value")
        controls.append(mode_dd)

    z_sl = widgets.FloatSlider(value=state["z_scale"], min=0.1, max=1.5, step=0.05, description="Z Height:")
    z_sl.observe(lambda ch: (state.update({"z_scale": ch["new"]}), _refresh()), names="value")
    controls.append(z_sl)

    cb_bg = widgets.Checkbox(value=(basemap == "esri"), description="3D Satellite Map")
    cb_bg.observe(lambda ch: (state.update({"basemap": "esri" if ch["new"] else None}), _refresh()), names="value")
    controls.append(cb_bg)

    box = widgets.VBox([widgets.HBox(controls), status, view_w])
    display(box)
    return box
