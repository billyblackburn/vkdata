"""Panel-style reactive widgets for vkpanel (backed by ipywidgets)."""

from __future__ import annotations

import html
from typing import Any, Callable, List, Optional, Sequence
import ipywidgets as _w


class _WidgetBase:
    """Thin Panel-style wrapper around an ipywidgets widget with .value and .on_change()."""

    def __init__(self, widget: Any) -> None:
        self._widget = widget

    @property
    def value(self) -> Any:
        return self._widget.value

    @value.setter
    def value(self, v: Any) -> None:
        self._widget.value = v

    def on_change(self, callback: Callable[[Any], None]) -> None:
        self._widget.observe(lambda ch: callback(ch["new"]), names="value")

    def to_widget(self) -> Any:
        return self._widget

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        return self._widget._repr_mimebundle_(**kwargs)


class FloatSlider(_WidgetBase):
    def __init__(
        self,
        name: str = "",
        start: float = 0.0,
        end: float = 1.0,
        step: float = 0.05,
        value: Optional[float] = None,
    ) -> None:
        v = start if value is None else value
        super().__init__(_w.FloatSlider(value=v, min=start, max=end, step=step, description=name))


class IntSlider(_WidgetBase):
    def __init__(
        self,
        name: str = "",
        start: int = 0,
        end: int = 10,
        step: int = 1,
        value: Optional[int] = None,
    ) -> None:
        v = start if value is None else value
        super().__init__(_w.IntSlider(value=v, min=start, max=end, step=step, description=name))


class Select(_WidgetBase):
    def __init__(self, name: str = "", options: Sequence[Any] = (), value: Any = None) -> None:
        v = options[0] if (value is None and options) else value
        super().__init__(_w.Dropdown(options=list(options), value=v, description=name))


class Checkbox(_WidgetBase):
    def __init__(self, name: str = "", value: bool = False) -> None:
        super().__init__(_w.Checkbox(value=bool(value), description=name))


class TextInput(_WidgetBase):
    def __init__(self, name: str = "", value: str = "", placeholder: str = "") -> None:
        super().__init__(_w.Text(value=value, placeholder=placeholder, description=name))


class Button(_WidgetBase):
    def __init__(self, name: str = "Click", button_type: str = "primary") -> None:
        super().__init__(_w.Button(description=name, button_style=button_type))

    def on_click(self, callback: Callable[[Any], None]) -> None:
        self._widget.on_click(callback)


def dataframe_to_html(df: Any, max_rows: int = 15) -> str:
    """Render a vkpandas.DataFrame (or dict) as a sleek dark-themed HTML table."""
    if hasattr(df, "head") and hasattr(df, "columns"):
        n_total = len(df)
        sub = df.head(max_rows)
        cols = list(sub.columns)
        col_vals = {c: sub[c].to_list() for c in cols}
        n_show = len(sub)
    elif isinstance(df, dict):
        cols = list(df.keys())
        col_vals = {c: list(v)[:max_rows] for c, v in df.items()}
        n_total = len(next(iter(df.values()), []))
        n_show = min(n_total, max_rows)
    else:
        return f"<pre>{html.escape(str(df))}</pre>"

    ths = "".join(f"<th style='padding:6px 10px;border-bottom:1px solid #3a4052;text-align:right'>{html.escape(str(c))}</th>" for c in cols)
    rows_html: List[str] = []
    for i in range(n_show):
        tds = []
        for c in cols:
            v = col_vals[c][i]
            s = f"{v:.3f}" if isinstance(v, float) else str(v)
            tds.append(f"<td style='padding:4px 10px;border-bottom:1px solid #262b38;text-align:right'>{html.escape(s)}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    footer = f"<div style='padding:4px 10px;color:#8c96aa;font-size:11px'>{n_total:,} rows x {len(cols)} cols</div>" if n_total > n_show else ""
    return (
        "<div style='background:#181b24;color:#e5e9f0;border:1px solid #3a4052;border-radius:6px;overflow-x:auto;font-family:monospace;font-size:12px'>"
        f"<table style='border-collapse:collapse;width:100%'><thead><tr>{ths}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>{footer}</div>"
    )


class DataTable(_WidgetBase):
    """Dark-themed interactive table widget for vkpandas DataFrames."""

    def __init__(self, df: Any = None, max_rows: int = 15) -> None:
        self.max_rows = int(max_rows)
        super().__init__(_w.HTML(value=dataframe_to_html(df, max_rows=self.max_rows) if df is not None else ""))

    def update(self, df: Any) -> None:
        self._widget.value = dataframe_to_html(df, max_rows=self.max_rows)
