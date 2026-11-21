"""Panel-style reactive layouts (Row, Column, Card, Tabs, Markdown, Pane) and bind()."""

from __future__ import annotations

import base64
import io
from typing import Any, Callable, Dict, List, Optional, Tuple
import ipywidgets as _w
from PIL import Image

from .widgets import _WidgetBase, dataframe_to_html


def _obj_to_html(obj: Any) -> str:
    """Convert any vkdata object (vkholoviews, vkshader, PIL.Image, vkpandas.DataFrame, str) to HTML."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "render") and callable(obj.render):
        obj = obj.render()
    if hasattr(obj, "to_pil") and callable(obj.to_pil):
        obj = obj.to_pil()
    if isinstance(obj, Image.Image):
        buf = io.BytesIO()
        obj.convert("RGB").save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"<img src='data:image/jpeg;base64,{b64}' style='max-width:100%;border-radius:4px;display:block'/>"
    if hasattr(obj, "columns") and hasattr(obj, "head"):
        return dataframe_to_html(obj)
    return f"<pre style='color:#e5e9f0;background:#181b24;padding:8px;border-radius:4px'>{obj}</pre>"


def to_ipywidget(obj: Any) -> _w.Widget:
    if isinstance(obj, _w.Widget):
        return obj
    if hasattr(obj, "to_widget") and callable(obj.to_widget):
        return obj.to_widget()
    return _w.HTML(value=_obj_to_html(obj))


class _LayoutBase:
    def __init__(self, widget: _w.Widget) -> None:
        self._widget = widget

    def to_widget(self) -> _w.Widget:
        return self._widget

    def show(self) -> Any:
        from IPython.display import display
        display(self._widget)
        return self

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        return self._widget._repr_mimebundle_(**kwargs)


class Row(_LayoutBase):
    def __init__(self, *items: Any) -> None:
        super().__init__(_w.HBox([to_ipywidget(it) for it in items]))


class Column(_LayoutBase):
    def __init__(self, *items: Any) -> None:
        super().__init__(_w.VBox([to_ipywidget(it) for it in items]))


class Card(_LayoutBase):
    def __init__(self, *items: Any, title: str = "") -> None:
        header = _w.HTML(
            value=f"<div style='font-weight:bold;color:#ff9128;padding:4px 8px;background:#1d212c;border-bottom:1px solid #3a4052'>{title}</div>"
        ) if title else None
        children = ([header] if header else []) + [to_ipywidget(it) for it in items]
        box = _w.VBox(children, layout=_w.Layout(border="1px solid #3a4052", padding="4px", margin="4px"))
        super().__init__(box)


class Tabs(_LayoutBase):
    def __init__(self, *named_items: Tuple[str, Any]) -> None:
        tab = _w.Tab(children=[to_ipywidget(it) for _, it in named_items])
        for i, (title, _) in enumerate(named_items):
            tab.set_title(i, str(title))
        super().__init__(tab)


class Markdown(_LayoutBase):
    def __init__(self, text: str) -> None:
        super().__init__(_w.HTML(value=f"<div style='color:#e5e9f0;font-family:sans-serif'>{text}</div>"))


class Pane(_LayoutBase):
    """Reactive display pane that can be updated in-place with any vkdata object."""

    def __init__(self, obj: Any = None) -> None:
        self._html = _w.HTML(value=_obj_to_html(obj))
        super().__init__(self._html)

    def update(self, obj: Any) -> None:
        self._html.value = _obj_to_html(obj)


class BoundPane(_LayoutBase):
    """Reactive pane returned by vkpanel.bind(fn, *widgets, **kw_widgets)."""

    def __init__(self, fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._html = _w.HTML()
        super().__init__(self._html)

        # Observe all bound widgets
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, _WidgetBase):
                a.on_change(lambda _: self.refresh())
            elif isinstance(a, _w.Widget) and a.has_trait("value"):
                a.observe(lambda _: self.refresh(), names="value")

        self.refresh()

    def _eval_arg(self, a: Any) -> Any:
        if isinstance(a, _WidgetBase):
            return a.value
        if isinstance(a, _w.Widget) and a.has_trait("value"):
            return a.value
        return a

    def refresh(self) -> None:
        ea = [self._eval_arg(a) for a in self._args]
        ekw = {k: self._eval_arg(v) for k, v in self._kwargs.items()}
        res = self._fn(*ea, **ekw)
        self._html.value = _obj_to_html(res)


def bind(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> BoundPane:
    """Bind a Python function to vkpanel widgets (exact Panel `pn.bind` pattern)."""
    return BoundPane(fn, args, kwargs)
