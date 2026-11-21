"""I/O and concatenation helpers for vkpandas."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import numpy as _np
import vknumpy as vnp

from .dataframe import DataFrame
from .series import Series


def concat(objs: Sequence[DataFrame], axis: int = 0) -> DataFrame:
    """Concatenate DataFrames vertically (axis=0) on GPU via vknumpy.concatenate."""
    if not objs:
        return DataFrame()
    if axis != 0:
        raise NotImplementedError("Only axis=0 vertical concat is currently supported")

    cols = objs[0].columns
    out: Dict[str, Series] = {}
    for c in cols:
        first = objs[0][c]
        if first.is_string:
            merged_np = _np.concatenate([df[c].to_numpy() for df in objs])
            out[c] = Series(merged_np, name=c)
        else:
            arr = vnp.concatenate([df[c]._data for df in objs], axis=0)
            out[c] = Series._from_vk(arr, name=c)
    return DataFrame(out)


def _parse_column(raw_vals: List[str]) -> Any:
    # Try int32 -> float32 -> string
    has_missing = False
    ints: List[int] = []
    all_int = True
    for s in raw_vals:
        s_strip = s.strip()
        if s_strip == "" or s_strip.lower() in ("nan", "na", "null", "none"):
            has_missing = True
            all_int = False
            break
        try:
            ints.append(int(s_strip))
        except ValueError:
            all_int = False
            break
    if all_int and not has_missing:
        return _np.array(ints, dtype=_np.int32)

    floats: List[float] = []
    all_float = True
    for s in raw_vals:
        s_strip = s.strip()
        if s_strip == "" or s_strip.lower() in ("nan", "na", "null", "none"):
            floats.append(_np.nan)
        else:
            try:
                floats.append(float(s_strip))
            except ValueError:
                all_float = False
                break
    if all_float:
        return _np.array(floats, dtype=_np.float32)

    return _np.array(raw_vals, dtype=object)


def read_csv(path: Union[str, Path], delimiter: str = ",") -> DataFrame:
    """Load a CSV file directly into GPU-backed vkpandas columns."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            return DataFrame()
        cols: Dict[str, List[str]] = {h.strip(): [] for h in header}
        keys = list(cols.keys())
        for row in reader:
            if not row:
                continue
            for k, val in zip(keys, row):
                cols[k].append(val)

    return DataFrame({k: _parse_column(v) for k, v in cols.items()})


def read_parquet(path: Union[str, Path], columns: Any = None) -> DataFrame:
    """Load a Parquet file into GPU-backed vkpandas columns via pyarrow + vknumpy.from_arrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required for read_parquet(); install it via `pixi add pyarrow` or `pip install pyarrow`") from exc

    table = pq.read_table(path, columns=columns)
    out: Dict[str, Series] = {}
    for name in table.column_names:
        converted = vnp.from_arrow(table[name])
        if isinstance(converted, tuple):
            codes, uniques = converted
            out[name] = Series._from_vk(codes, name=name, categories=uniques)
        else:
            out[name] = Series._from_vk(converted, name=name)
    return DataFrame(out)
