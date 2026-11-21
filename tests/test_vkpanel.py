import vkpandas as vpd
import vkpanel as pn


def test_vkpanel_widgets_layout_and_reactive_bind():
    s = pn.FloatSlider(name="Scale", start=1.0, end=10.0, value=2.0)
    sel = pn.Select(name="Mode", options=["sum", "mean"], value="sum")

    calls = []

    def compute(scale: float, mode: str) -> str:
        calls.append((scale, mode))
        return f"<b>{mode}:{scale:.1f}</b>"

    bound = pn.bind(compute, scale=s, mode=sel)
    assert len(calls) == 1
    assert "sum:2.0" in bound._html.value

    # Changing widget triggers reactive recompute
    s.value = 5.0
    assert len(calls) == 2
    assert "sum:5.0" in bound._html.value

    sel.value = "mean"
    assert len(calls) == 3
    assert "mean:5.0" in bound._html.value

    df = vpd.DataFrame({"city": ["nyc", "la"], "val": [10.0, 20.0]})
    tbl = pn.DataTable(df)
    assert "nyc" in tbl._widget.value and "20.000" in tbl._widget.value

    dash = pn.Column(
        pn.Row(s, sel),
        pn.Card(bound, tbl, title="GPU Dashboard"),
        pn.Tabs(("Summary", bound), ("Table", tbl)),
    )
    assert dash.to_widget() is not None
