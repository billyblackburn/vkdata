"""GPU-backed columnar DataFrame and GroupBy for vkpandas."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as _np
import vknumpy as vnp

from .series import Series


class GroupBy:
    """GPU-accelerated GroupBy over a vkpandas DataFrame using vknumpy.groupby_agg."""

    def __init__(self, df: "DataFrame", by: str) -> None:
        if by not in df._columns:
            raise KeyError(f"Column {by!r} not found in DataFrame")
        self._df = df
        self._by = by

    def _agg_all(self, op: str) -> "DataFrame":
        key_col = self._df._columns[self._by]
        out_cols: Dict[str, Series] = {}
        u_keys_series: Optional[Series] = None

        for col_name, col in self._df._columns.items():
            if col_name == self._by:
                continue
            if op != "count" and col.is_string:
                continue
            uk, res = vnp.groupby_agg(key_col._data, col._data, op=op)
            if u_keys_series is None:
                u_keys_series = Series._from_vk(uk, name=self._by, categories=key_col._categories)
                out_cols[self._by] = u_keys_series
            out_cols[col_name] = Series._from_vk(res, name=col_name)

        if u_keys_series is None:
            uk, cnt = vnp.unique(key_col._data, return_counts=True)
            out_cols[self._by] = Series._from_vk(uk, name=self._by, categories=key_col._categories)
            if op == "count":
                out_cols["count"] = Series._from_vk(cnt, name="count")

        return DataFrame(out_cols)

    def sum(self) -> "DataFrame":
        return self._agg_all("sum")

    def mean(self) -> "DataFrame":
        return self._agg_all("mean")

    def count(self) -> "DataFrame":
        return self._agg_all("count")

    def var(self) -> "DataFrame":
        return self._agg_all("var")

    def std(self) -> "DataFrame":
        return self._agg_all("std")

    def agg(self, spec: Dict[str, str]) -> "DataFrame":
        key_col = self._df._columns[self._by]
        out_cols: Dict[str, Series] = {}
        u_keys_series: Optional[Series] = None

        for col_name, op in spec.items():
            if col_name not in self._df._columns:
                raise KeyError(f"Column {col_name!r} not found in DataFrame")
            col = self._df._columns[col_name]
            uk, res = vnp.groupby_agg(key_col._data, col._data, op=op)
            if u_keys_series is None:
                u_keys_series = Series._from_vk(uk, name=self._by, categories=key_col._categories)
                out_cols[self._by] = u_keys_series
            out_cols[col_name] = Series._from_vk(res, name=col_name)

        return DataFrame(out_cols)


class DataFrame:
    """Columnar Vulkan GPU DataFrame built on top of vknumpy."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._columns: Dict[str, Series] = {}
        if data is None:
            return
        expected_len: Optional[int] = None
        for col_name, col_data in data.items():
            s = col_data if isinstance(col_data, Series) else Series(col_data, name=col_name)
            if s.name != col_name:
                s = Series._from_vk(s._data, name=col_name, categories=s._categories)
            if expected_len is None:
                expected_len = len(s)
            elif len(s) != expected_len:
                raise ValueError(
                    f"All columns must have the same length: {col_name!r} has {len(s)}, expected {expected_len}"
                )
            self._columns[col_name] = s

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def columns(self) -> List[str]:
        return list(self._columns.keys())

    @property
    def dtypes(self) -> Dict[str, Any]:
        return {name: col.dtype for name, col in self._columns.items()}

    @property
    def shape(self) -> Tuple[int, int]:
        n_rows = len(next(iter(self._columns.values()))) if self._columns else 0
        return (n_rows, len(self._columns))

    def __len__(self) -> int:
        return self.shape[0]

    def copy(self) -> "DataFrame":
        return DataFrame({name: col.copy() for name, col in self._columns.items()})

    def head(self, n: int = 5) -> "DataFrame":
        return DataFrame({name: col.head(n) for name, col in self._columns.items()})

    def tail(self, n: int = 5) -> "DataFrame":
        return DataFrame({name: col.tail(n) for name, col in self._columns.items()})

    def show(self, n: int = 10) -> None:
        """Print a formatted preview of the first n rows of the DataFrame."""
        n_rows, n_cols = self.shape
        if n_cols == 0:
            print("Empty DataFrame")
            return
        preview = self.head(n)
        cols = preview.columns
        col_data = {c: [str(v) for v in preview._columns[c].to_list()] for c in cols}
        widths = {c: max(len(c), *(len(s) for s in col_data[c]), 1) for c in cols}
        header = "    " + "  ".join(c.rjust(widths[c]) for c in cols)
        rows = [
            f"{i:<3d} " + "  ".join(col_data[c][i].rjust(widths[c]) for c in cols)
            for i in range(len(preview))
        ]
        if n_rows > n:
            rows.append(f"... ({n_rows} rows x {n_cols} columns)")
        print("\n".join([header] + rows))

    def to_dict(self) -> Dict[str, List[Any]]:
        return {name: col.to_list() for name, col in self._columns.items()}

    def to_pandas(self) -> Any:
        import pandas as pd
        return pd.DataFrame({name: col.to_numpy() for name, col in self._columns.items()})

    def __repr__(self) -> str:
        n_rows, n_cols = self.shape
        if n_cols == 0:
            return "Empty DataFrame"
        preview = self.head(10)
        cols = preview.columns
        col_data = {c: [str(v) for v in preview._columns[c].to_list()] for c in cols}
        widths = {c: max(len(c), *(len(s) for s in col_data[c]), 1) for c in cols}
        header = "    " + "  ".join(c.rjust(widths[c]) for c in cols)
        rows = []
        for i in range(len(preview)):
            row_str = f"{i:<3d} " + "  ".join(col_data[c][i].rjust(widths[c]) for c in cols)
            rows.append(row_str)
        if n_rows > 10:
            rows.append(f"... ({n_rows} rows x {n_cols} columns)")
        return "\n".join([header] + rows)

    # ------------------------------------------------------------------
    # Indexing, Filtering & Column Management
    # ------------------------------------------------------------------
    def _take_rows(self, idx: vnp.ndarray) -> "DataFrame":
        return DataFrame({name: col._take(idx) for name, col in self._columns.items()})

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._columns[key]
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            return DataFrame({k: self._columns[k] for k in key})
        if isinstance(key, Series):
            key = key._data
        if isinstance(key, vnp.ndarray) and key.dtype == _np.bool_:
            idx = vnp.flatnonzero(key)
            return self._take_rows(idx)
        if isinstance(key, vnp.ndarray) and _np.issubdtype(key.dtype, _np.integer):
            return self._take_rows(key)
        if isinstance(key, slice):
            return DataFrame({name: col[key] for name, col in self._columns.items()})
        raise KeyError(f"Unsupported DataFrame key: {type(key)}")

    def __setitem__(self, key: str, value: Any) -> None:
        s = value if isinstance(value, Series) else Series(value, name=key)
        if self._columns and len(s) != len(self):
            raise ValueError(f"Column {key!r} has length {len(s)}, expected {len(self)}")
        self._columns[key] = Series._from_vk(s._data, name=key, categories=s._categories)

    def drop(self, columns: Union[str, Sequence[str]]) -> "DataFrame":
        drop_set = {columns} if isinstance(columns, str) else set(columns)
        return DataFrame({k: v for k, v in self._columns.items() if k not in drop_set})

    # ------------------------------------------------------------------
    # Missing values, Sorting, GroupBy, Merge
    # ------------------------------------------------------------------
    def fillna(self, value: float) -> "DataFrame":
        return DataFrame({name: col.fillna(value) for name, col in self._columns.items()})

    def dropna(self, subset: Optional[Sequence[str]] = None) -> "DataFrame":
        cols = list(subset) if subset is not None else self.columns
        valid_mask: Optional[vnp.ndarray] = None
        for c in cols:
            col = self._columns[c]
            m = col.notna()._data
            valid_mask = m if valid_mask is None else vnp.logical_and(valid_mask, m)
        if valid_mask is None:
            return self.copy()
        return self[valid_mask]

    def sort_values(self, by: Union[str, Sequence[str]], ascending: bool = True) -> "DataFrame":
        if isinstance(by, str):
            col = self._columns[by]
            order = vnp.argsort(col._data, descending=not ascending)
            return self._take_rows(order)

        by_list = list(by)
        if len(by_list) == 1:
            return self.sort_values(by_list[0], ascending=ascending)

        # numpy.lexsort uses last key as primary -> reverse by_list
        keys = [self._columns[c]._data for c in reversed(by_list)]
        order = vnp.lexsort(keys)
        if not ascending:
            order = vnp.flip(order)
        return self._take_rows(order)

    def groupby(self, by: str) -> GroupBy:
        return GroupBy(self, by)

    def merge(
        self,
        right: "DataFrame",
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = "inner",
        suffixes: Tuple[str, str] = ("_x", "_y"),
    ) -> "DataFrame":
        """GPU inner join using vknumpy.merge_indices (right key must be unique)."""
        if how != "inner":
            raise NotImplementedError(f"how={how!r} not yet implemented; currently supports how='inner'")
        lk_name = on if on is not None else left_on
        rk_name = on if on is not None else right_on
        if lk_name is None or rk_name is None:
            raise ValueError("Must specify `on` or both `left_on` and `right_on`")

        lk_col = self._columns[lk_name]
        rk_col = right._columns[rk_name]

        if lk_col.is_string or rk_col.is_string:
            # Align string dictionaries onto shared int32 codes
            combined = _np.concatenate([lk_col.to_numpy(), rk_col.to_numpy()])
            codes, _ = vnp.factorize(combined)
            l_codes = codes[: len(lk_col)]
            r_codes = codes[len(lk_col) :]
            l_idx, r_idx = vnp.merge_indices(l_codes, r_codes)
        else:
            l_idx, r_idx = vnp.merge_indices(lk_col._data, rk_col._data)

        out_cols: Dict[str, Series] = {}
        for name, col in self._columns.items():
            out_name = name
            if name in right._columns and name != rk_name:
                out_name = f"{name}{suffixes[0]}"
            out_cols[out_name] = col._take(l_idx)

        for name, col in right._columns.items():
            if name == rk_name and lk_name == rk_name:
                continue
            out_name = name
            if name in self._columns:
                out_name = f"{name}{suffixes[1]}"
            out_cols[out_name] = col._take(r_idx)

        return DataFrame(out_cols)

    def describe(self, *, max_categories: int = 8) -> str:
        """Return a readable GPU-computed summary of ranges, NaNs, and categorical counts."""
        n_rows, n_cols = self.shape
        col_w = max((len(c) for c in self._columns), default=8)
        lines = [f"vkpandas.DataFrame ({n_rows:,} rows x {n_cols} columns)"]
        for name, s in self._columns.items():
            n_nan = int(s.isna().sum())
            if s.is_string:
                vc = s.value_counts()
                items = list(vc.items())
                shown = ", ".join(f"{k}({int(v)})" for k, v in items[:max_categories])
                if len(items) > max_categories:
                    shown += f", ... (+{len(items) - max_categories} more)"
                lines.append(f"  {name:{col_w}s} | CATEGORICAL ({len(items)} unique): {shown}")
            else:
                clean = s.dropna() if n_nan else s
                if len(clean) == 0:
                    lines.append(f"  {name:{col_w}s} | ALL NaN (NaNs={n_nan})")
                else:
                    lines.append(
                        f"  {name:{col_w}s} | min={float(clean.min()):10.2f}  "
                        f"mean={float(clean.mean()):10.2f}  std={float(clean.std()):9.2f}  "
                        f"max={float(clean.max()):10.2f}  NaNs={n_nan}"
                    )
        return "\n".join(lines)
