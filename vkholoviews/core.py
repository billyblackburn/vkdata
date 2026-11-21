"""Core declarative HoloViews-style primitives: Viewable, Element, Overlay (*), Layout (+)."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class Viewable:
    """Base class for all composable vkholoviews objects (Element, Overlay, Layout)."""

    def __init__(self, opts: Optional[Dict[str, Any]] = None) -> None:
        self._opts: Dict[str, Any] = dict(opts) if opts else {}

    def opts(self, **kwargs: Any) -> "Viewable":
        """Return a shallow copy with updated visual/rendering options."""
        clone = copy.copy(self)
        clone._opts = {**self._opts, **kwargs}
        return clone

    def get_opt(self, key: str, default: Any = None) -> Any:
        return self._opts.get(key, default)

    def __mul__(self, other: "Viewable") -> "Overlay":
        """Compose elements on top of each other (HoloViews Overlay syntax: A * B)."""
        layers: List[Viewable] = []
        if isinstance(self, Overlay):
            layers.extend(self.layers)
        else:
            layers.append(self)
        if isinstance(other, Overlay):
            layers.extend(other.layers)
        else:
            layers.append(other)
        merged_opts = {**self._opts, **other._opts}
        return Overlay(layers, opts=merged_opts)

    def __add__(self, other: "Viewable") -> "Layout":
        """Compose elements side-by-side / in a grid (HoloViews Layout syntax: A + B)."""
        items: List[Viewable] = []
        if isinstance(self, Layout):
            items.extend(self.items)
        else:
            items.append(self)
        if isinstance(other, Layout):
            items.extend(other.items)
        else:
            items.append(other)
        return Layout(items)

    def render(self) -> Any:
        """Render this Viewable to a PIL.Image.Image."""
        from .render import render_viewable
        return render_viewable(self)

    def save(self, path: Union[str, Path], **kwargs: Any) -> Path:
        p = Path(path)
        img = self.render()
        if p.suffix.lower() in (".jpg", ".jpeg"):
            img.convert("RGB").save(p, format="JPEG", **kwargs)
        else:
            img.save(p, **kwargs)
        return p

    def interact(self, **kwargs: Any) -> Any:
        """Launch an interactive Jupyter pan/zoom/shade widget for this Viewable."""
        from .render import interact_viewable
        return interact_viewable(self, **kwargs)

    def _repr_png_(self) -> bytes:
        buf = io.BytesIO()
        self.render().save(buf, format="PNG", compress_level=1)
        return buf.getvalue()


class Element(Viewable):
    """Base declarative data element wrapping GPU-resident vkpandas / vknumpy data."""

    def __init__(
        self,
        data: Any,
        kdims: Optional[Union[str, Sequence[str]]] = None,
        vdims: Optional[Union[str, Sequence[str]]] = None,
        opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(opts=opts)
        self.data = data
        self.kdims: List[str] = [kdims] if isinstance(kdims, str) else list(kdims or [])
        self.vdims: List[str] = [vdims] if isinstance(vdims, str) else list(vdims or [])


class Overlay(Viewable):
    """Layered composition (A * B) sharing a single coordinate space."""

    def __init__(self, layers: Sequence[Viewable], opts: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(opts=opts)
        self.layers: List[Viewable] = list(layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        return " * ".join(type(l).__name__ for l in self.layers)


class Layout(Viewable):
    """Multi-panel grid composition (A + B + C)."""

    def __init__(
        self,
        items: Sequence[Viewable],
        cols: Optional[int] = None,
        opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(opts=opts)
        self.items: List[Viewable] = list(items)
        self._cols = cols

    def cols(self, n: int) -> "Layout":
        clone = copy.copy(self)
        clone._cols = int(n)
        return clone

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return " + ".join(repr(it) for it in self.items)
