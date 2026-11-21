"""Interactive mouse/wheel/touch/box-zoom anywidget for vkshader & vkholoviews."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import anywidget
import traitlets

_ESM = r"""
function render({ model, el }) {
  const w = model.get("width");
  const h = model.get("height");

  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.width = `${w}px`;
  wrap.style.height = `${h}px`;
  wrap.style.overflow = "hidden";
  wrap.style.background = "#12141a";
  wrap.style.border = "1px solid #3a4052";
  wrap.style.borderRadius = "4px";
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

  const box = document.createElement("div");
  box.style.position = "absolute";
  box.style.border = "2px solid #ff9128";
  box.style.background = "rgba(255, 145, 40, 0.16)";
  box.style.display = "none";
  box.style.pointerEvents = "none";
  wrap.appendChild(box);

  const hint = document.createElement("div");
  hint.textContent = "Drag: pan | Wheel: zoom | Shift+Drag: box zoom | Dbl-click: +2x | Right-click: -2x";
  hint.style.position = "absolute";
  hint.style.left = "8px";
  hint.style.bottom = "6px";
  hint.style.padding = "2px 6px";
  hint.style.fontSize = "11px";
  hint.style.fontFamily = "monospace";
  hint.style.color = "#e5e9f0";
  hint.style.background = "rgba(15, 18, 24, 0.72)";
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

  // Send viewport change to Python
  let lastSend = 0;
  function setViewport(xr0, xr1, yr0, yr1, force = false) {
    const now = performance.now();
    if (!force && now - lastSend < 45) return;
    lastSend = now;
    model.set("viewport", [xr0, xr1, yr0, yr1]);
    model.save_changes();
  }

  function zoomAt(u, v, scale) {
    const [xr0, xr1, yr0, yr1] = model.get("viewport");
    const cx = xr0 + u * (xr1 - xr0);
    const cy = yr0 + v * (yr1 - yr0);
    const nx0 = cx - u * (xr1 - xr0) * scale;
    const nx1 = cx + (1.0 - u) * (xr1 - xr0) * scale;
    const ny0 = cy - v * (yr1 - yr0) * scale;
    const ny1 = cy + (1.0 - v) * (yr1 - yr0) * scale;
    setViewport(nx0, nx1, ny0, ny1, true);
  }

  let dragging = false;
  let boxMode = false;
  let startX = 0, startY = 0;
  let startVp = null;

  wrap.addEventListener("pointerdown", (e) => {
    if (e.button === 2) return; // right click handled by contextmenu
    e.preventDefault();
    wrap.setPointerCapture(e.pointerId);
    const rect = wrap.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    startVp = [...model.get("viewport")];
    dragging = true;
    boxMode = e.shiftKey;
    wrap.style.cursor = boxMode ? "crosshair" : "grabbing";
    if (boxMode) {
      box.style.left = `${startX}px`;
      box.style.top = `${startY}px`;
      box.style.width = "0px";
      box.style.height = "0px";
      box.style.display = "block";
    }
  });

  wrap.addEventListener("pointermove", (e) => {
    if (!dragging || !startVp) return;
    const rect = wrap.getBoundingClientRect();
    const cx = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const cy = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

    if (boxMode) {
      const lx = Math.min(startX, cx);
      const ty = Math.min(startY, cy);
      box.style.left = `${lx}px`;
      box.style.top = `${ty}px`;
      box.style.width = `${Math.abs(cx - startX)}px`;
      box.style.height = `${Math.abs(cy - startY)}px`;
    } else {
      const dxPx = cx - startX;
      const dyPx = cy - startY;
      const [xr0, xr1, yr0, yr1] = startVp;
      const dx = (dxPx / rect.width) * (xr1 - xr0);
      const dy = (dyPx / rect.height) * (yr1 - yr0);
      setViewport(xr0 - dx, xr1 - dx, yr0 + dy, yr1 + dy, false);
    }
  });

  function endDrag(e) {
    if (!dragging || !startVp) return;
    const rect = wrap.getBoundingClientRect();
    const cx = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const cy = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    dragging = false;
    wrap.style.cursor = "grab";

    if (boxMode) {
      box.style.display = "none";
      boxMode = false;
      const dxPx = Math.abs(cx - startX);
      const dyPx = Math.abs(cy - startY);
      if (dxPx > 6 && dyPx > 6) {
        const [xr0, xr1, yr0, yr1] = startVp;
        const u0 = Math.min(startX, cx) / rect.width;
        const u1 = Math.max(startX, cx) / rect.width;
        const v1 = 1.0 - Math.min(startY, cy) / rect.height;
        const v0 = 1.0 - Math.max(startY, cy) / rect.height;
        setViewport(
          xr0 + u0 * (xr1 - xr0),
          xr0 + u1 * (xr1 - xr0),
          yr0 + v0 * (yr1 - yr0),
          yr0 + v1 * (yr1 - yr0),
          true
        );
      }
    } else {
      const dxPx = cx - startX;
      const dyPx = cy - startY;
      if (Math.abs(dxPx) > 1 || Math.abs(dyPx) > 1) {
        const [xr0, xr1, yr0, yr1] = startVp;
        const dx = (dxPx / rect.width) * (xr1 - xr0);
        const dy = (dyPx / rect.height) * (yr1 - yr0);
        setViewport(xr0 - dx, xr1 - dx, yr0 + dy, yr1 + dy, true);
      }
    }
    startVp = null;
  }

  wrap.addEventListener("pointerup", endDrag);
  wrap.addEventListener("pointercancel", endDrag);

  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const v = Math.max(0, Math.min(1, 1.0 - (e.clientY - rect.top) / rect.height));
    const scale = e.deltaY < 0 ? 0.72 : 1.0 / 0.72;
    zoomAt(u, v, scale);
  }, { passive: false });

  wrap.addEventListener("dblclick", (e) => {
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const u = (e.clientX - rect.left) / rect.width;
    const v = 1.0 - (e.clientY - rect.top) / rect.height;
    zoomAt(u, v, 0.5);
  });

  wrap.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const u = (e.clientX - rect.left) / rect.width;
    const v = 1.0 - (e.clientY - rect.top) / rect.height;
    zoomAt(u, v, 2.0);
  });

  el.appendChild(wrap);
  return () => {
    if (prevUrl) URL.revokeObjectURL(prevUrl);
  };
}
export default { render };
"""


class InteractiveMapWidget(anywidget.AnyWidget):
    """Mouse-drag pan, cursor-centered wheel zoom, Shift+drag box-zoom, and touch map widget."""

    _esm = _ESM
    width = traitlets.Int(640).tag(sync=True)
    height = traitlets.Int(640).tag(sync=True)
    viewport = traitlets.List(traitlets.Float(), default_value=[-8242000.0, -8226000.0, 4968000.0, 4985000.0]).tag(sync=True)
    image_bytes = traitlets.Bytes(b"").tag(sync=True)

    def __init__(
        self,
        width: int,
        height: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        image_bytes: bytes = b"",
        on_viewport_change: Optional[Callable[[Tuple[float, float], Tuple[float, float]], None]] = None,
    ) -> None:
        super().__init__(
            width=int(width),
            height=int(height),
            viewport=[float(x_range[0]), float(x_range[1]), float(y_range[0]), float(y_range[1])],
            image_bytes=image_bytes,
        )
        self._callback = on_viewport_change
        self.observe(self._handle_vp, names="viewport")

    def _handle_vp(self, change: dict) -> None:
        if self._callback is None:
            return
        vp = change["new"]
        if len(vp) == 4:
            self._callback((float(vp[0]), float(vp[1])), (float(vp[2]), float(vp[3])))
