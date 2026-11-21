"""GPU-backed Series class for vkpandas (numeric + dictionary-encoded strings)."""

from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as _np
import vknumpy as vnp


class Rolling:
    """Rolling window view over a vkpandas Series (backed by GPU cumsum)."""

    def __init__(self, series: "Series", window: int) -> None:
        self._series = series
        self._window = int(window)

    def sum(self) -> "Series":
        return Series._from_vk(vnp.rolling_sum(self._series._data, self._window), name=self._series.name)

    def mean(self) -> "Series":
        return Series._from_vk(vnp.rolling_mean(self._series._data, self._window), name=self._series.name)


class Series:
    """1D GPU-backed columnar Series (numeric float32/int32/bool_ or dictionary-encoded strings)."""

    __slots__ = ("_data", "_categories", "name")

    def __init__(
        self,
        data: Any,
        name: Optional[str] = None,
        dtype: Any = None,
    ) -> None:
        self.name = name
        self._categories: Optional[_np.ndarray] = None

        if isinstance(data, Series):
            self._data = data._data.copy()
            self._categories = None if data._categories is None else data._categories.copy()
            if name is None:
                self.name = data.name
            return

        if isinstance(data, vnp.ndarray):
            self._data = data.ravel() if dtype is None else data.ravel().astype(dtype)
            return

        arr = _np.asarray(data)
        if arr.dtype.kind in ("U", "S", "O") and dtype is None:
            codes, uniques = vnp.factorize(arr)
            self._data = codes
            self._categories = uniques
        else:
            self._data = vnp.array(arr, dtype=dtype).ravel()

    @classmethod
    def _from_vk(
        cls,
        arr: vnp.ndarray,
        name: Optional[str] = None,
        categories: Optional[_np.ndarray] = None,
    ) -> "Series":
        obj = cls.__new__(cls)
        obj._data = arr.ravel()
        obj._categories = categories
        obj.name = name
        return obj

    # ------------------------------------------------------------------
    # Properties & conversions
    # ------------------------------------------------------------------
    @property
    def is_string(self) -> bool:
        return self._categories is not None

    @property
    def dtype(self) -> Any:
        return "string" if self.is_string else self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def values(self) -> vnp.ndarray:
        return self._data

    def __len__(self) -> int:
        return self._data.shape[0]

    def to_numpy(self) -> _np.ndarray:
        raw = self._data.to_numpy()
        if self._categories is not None:
            return self._categories[raw]
        return raw

    def to_list(self) -> list[Any]:
        return self.to_numpy().tolist()

    def head(self, n: int = 5) -> "Series":
        return Series._from_vk(self._data[:n], name=self.name, categories=self._categories)

    def tail(self, n: int = 5) -> "Series":
        start = max(0, len(self) - n)
        return Series._from_vk(self._data[start:], name=self.name, categories=self._categories)

    def show(self, n: int = 10) -> None:
        """Print a formatted preview of the first n elements of the Series."""
        vals = self.head(n).to_numpy()
        lines = [f"{i}\t{v}" for i, v in enumerate(vals)]
        if len(self) > n:
            lines.append(f"... ({len(self)} rows total)")
        footer = f"Name: {self.name}, dtype: {self.dtype}" if self.name else f"dtype: {self.dtype}"
        print("\n".join(lines + [footer]))

    def copy(self) -> "Series":
        return Series._from_vk(
            self._data.copy(),
            name=self.name,
            categories=None if self._categories is None else self._categories.copy(),
        )

    def astype(self, dtype: Any) -> "Series":
        if self.is_string:
            return Series(self.to_numpy().astype(dtype), name=self.name)
        return Series._from_vk(self._data.astype(dtype), name=self.name)

    def __repr__(self) -> str:
        vals = self.to_numpy()
        lines = [f"{i}\t{v}" for i, v in enumerate(vals[:10])]
        if len(vals) > 10:
            lines.append(f"... ({len(vals)} rows total)")
        footer = f"Name: {self.name}, dtype: {self.dtype}" if self.name else f"dtype: {self.dtype}"
        return "\n".join(lines + [footer])

    # ------------------------------------------------------------------
    # Indexing & Filtering (GPU stream compaction & take)
    # ------------------------------------------------------------------
    def _take(self, idx: vnp.ndarray) -> "Series":
        if idx.size == 0:
            return Series._from_vk(
                vnp.empty((0,), dtype=self._data.dtype),
                name=self.name,
                categories=self._categories,
            )
        return Series._from_vk(vnp.take(self._data, idx), name=self.name, categories=self._categories)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, Series):
            key = key._data
        if isinstance(key, vnp.ndarray) and key.dtype == _np.bool_:
            idx = vnp.flatnonzero(key)
            return self._take(idx)
        if isinstance(key, vnp.ndarray) and _np.issubdtype(key.dtype, _np.integer):
            return self._take(key)
        if isinstance(key, slice):
            return Series._from_vk(self._data[key], name=self.name, categories=self._categories)
        if isinstance(key, int):
            val = self._data[key].item()
            return self._categories[int(val)] if self._categories is not None else val
        return self._take(vnp.array(key, dtype=_np.int32))

    # ------------------------------------------------------------------
    # Operators (arithmetic, comparisons, logicals)
    # ------------------------------------------------------------------
    def _bin_other(self, other: Any) -> Any:
        if isinstance(other, Series):
            if self.is_string and other.is_string:
                # Re-align string codes if categories differ
                if _np.array_equal(self._categories, other._categories):
                    return other._data
                raise TypeError("String comparison between Series with different categories not yet aligned")
            return other._data
        if self.is_string and isinstance(other, str):
            matches = _np.where(self._categories == other)[0]
            return int(matches[0]) if len(matches) > 0 else -1
        return other

    def __add__(self, other: Any) -> "Series":
        return Series._from_vk(self._data + self._bin_other(other), name=self.name)

    def __radd__(self, other: Any) -> "Series":
        return Series._from_vk(self._bin_other(other) + self._data, name=self.name)

    def __sub__(self, other: Any) -> "Series":
        return Series._from_vk(self._data - self._bin_other(other), name=self.name)

    def __rsub__(self, other: Any) -> "Series":
        return Series._from_vk(self._bin_other(other) - self._data, name=self.name)

    def __mul__(self, other: Any) -> "Series":
        return Series._from_vk(self._data * self._bin_other(other), name=self.name)

    def __rmul__(self, other: Any) -> "Series":
        return Series._from_vk(self._bin_other(other) * self._data, name=self.name)

    def __truediv__(self, other: Any) -> "Series":
        return Series._from_vk(self._data / self._bin_other(other), name=self.name)

    def __rtruediv__(self, other: Any) -> "Series":
        return Series._from_vk(self._bin_other(other) / self._data, name=self.name)

    def __neg__(self) -> "Series":
        return Series._from_vk(-self._data, name=self.name)

    def __gt__(self, other: Any) -> "Series":
        return Series._from_vk(self._data > self._bin_other(other), name=self.name)

    def __lt__(self, other: Any) -> "Series":
        return Series._from_vk(self._data < self._bin_other(other), name=self.name)

    def __ge__(self, other: Any) -> "Series":
        return Series._from_vk(self._data >= self._bin_other(other), name=self.name)

    def __le__(self, other: Any) -> "Series":
        return Series._from_vk(self._data <= self._bin_other(other), name=self.name)

    def __eq__(self, other: Any) -> "Series":  # type: ignore[override]
        return Series._from_vk(self._data == self._bin_other(other), name=self.name)

    def __ne__(self, other: Any) -> "Series":  # type: ignore[override]
        return Series._from_vk(self._data != self._bin_other(other), name=self.name)

    def __and__(self, other: Any) -> "Series":
        return Series._from_vk(vnp.logical_and(self._data, self._bin_other(other)), name=self.name)

    def __or__(self, other: Any) -> "Series":
        return Series._from_vk(vnp.logical_or(self._data, self._bin_other(other)), name=self.name)

    def __invert__(self) -> "Series":
        return Series._from_vk(vnp.logical_not(self._data), name=self.name)

    # ------------------------------------------------------------------
    # Aggregations (skipna=True by default, matching pandas)
    # ------------------------------------------------------------------
    def sum(self, skipna: bool = True) -> float:
        return float((vnp.nansum(self._data) if skipna else vnp.sum(self._data)).item())

    def mean(self, skipna: bool = True) -> float:
        return float((vnp.nanmean(self._data) if skipna else vnp.mean(self._data)).item())

    def var(self, ddof: int = 1, skipna: bool = True) -> float:
        return float((vnp.nanvar(self._data, ddof=ddof) if skipna else vnp.var(self._data, ddof=ddof)).item())

    def std(self, ddof: int = 1, skipna: bool = True) -> float:
        return float((vnp.nanstd(self._data, ddof=ddof) if skipna else vnp.std(self._data, ddof=ddof)).item())

    def max(self, skipna: bool = True) -> float:
        return float((vnp.nanmax(self._data) if skipna else vnp.max(self._data)).item())

    def min(self, skipna: bool = True) -> float:
        return float((vnp.nanmin(self._data) if skipna else vnp.min(self._data)).item())

    def count(self) -> int:
        if self.is_string:
            return len(self)
        return int(vnp.count_nonzero(vnp.logical_not(vnp.isnan(self._data))).item())

    # ------------------------------------------------------------------
    # Missing values, transforms, sorting, unique, value_counts, rolling
    # ------------------------------------------------------------------
    def isna(self) -> "Series":
        if self.is_string:
            return Series._from_vk(vnp.zeros(self.shape, dtype=_np.bool_), name=self.name)
        return Series._from_vk(vnp.isnan(self._data), name=self.name)

    def notna(self) -> "Series":
        return ~self.isna()

    def fillna(self, value: float) -> "Series":
        if self.is_string:
            return self.copy()
        return Series._from_vk(vnp.nan_to_num(self._data, nan=float(value)), name=self.name)

    def dropna(self) -> "Series":
        return self[self.notna()]

    def ffill(self) -> "Series":
        return Series._from_vk(vnp.ffill(self._data), name=self.name, categories=self._categories)

    def bfill(self) -> "Series":
        return Series._from_vk(vnp.bfill(self._data), name=self.name, categories=self._categories)

    def cumsum(self) -> "Series":
        return Series._from_vk(vnp.cumsum(self._data), name=self.name)

    def diff(self, periods: int = 1) -> "Series":
        if periods != 1:
            return self - self.shift(periods)
        d = vnp.diff(self._data)
        prefix = vnp.full((1,), _np.nan, dtype=_np.float32)
        return Series._from_vk(vnp.concatenate([prefix, d.astype(_np.float32)], axis=0), name=self.name)

    def shift(self, periods: int = 1, fill_value: float = _np.nan) -> "Series":
        n = len(self)
        if periods == 0 or n == 0:
            return self.copy()
        af = self._data.astype(_np.float32)
        if abs(periods) >= n:
            return Series._from_vk(vnp.full((n,), fill_value, dtype=_np.float32), name=self.name)
        if periods > 0:
            pad = vnp.full((periods,), fill_value, dtype=_np.float32)
            return Series._from_vk(vnp.concatenate([pad, af[:-periods]], axis=0), name=self.name)
        pad = vnp.full((-periods,), fill_value, dtype=_np.float32)
        return Series._from_vk(vnp.concatenate([af[-periods:], pad], axis=0), name=self.name)

    def sort_values(self, ascending: bool = True) -> "Series":
        order = vnp.argsort(self._data, descending=not ascending)
        return self._take(order)

    def unique(self) -> "Series":
        u = vnp.unique(self._data)
        return Series._from_vk(u, name=self.name, categories=self._categories)

    def nunique(self) -> int:
        return len(self.unique())

    def value_counts(self, sort: bool = True) -> dict[Any, int]:
        u, counts = vnp.unique(self._data, return_counts=True)
        if sort:
            order = vnp.argsort(counts, descending=True)
            u = vnp.take(u, order)
            counts = vnp.take(counts, order)
        u_np = u.to_numpy()
        c_np = counts.to_numpy()
        if self._categories is not None:
            return {str(self._categories[int(k)]): int(c) for k, c in zip(u_np, c_np)}
        return {k.item(): int(c) for k, c in zip(u_np, c_np)}

    def isin(self, values: Sequence[Any]) -> "Series":
        if self.is_string:
            code_list = []
            for v in values:
                matches = _np.where(self._categories == v)[0]
                if len(matches) > 0:
                    code_list.append(int(matches[0]))
            if not code_list:
                return Series._from_vk(vnp.zeros(self.shape, dtype=_np.bool_), name=self.name)
            return Series._from_vk(vnp.isin(self._data, vnp.array(code_list, dtype=_np.int32)), name=self.name)
        return Series._from_vk(vnp.isin(self._data, vnp.asarray(values)), name=self.name)

    def rolling(self, window: int) -> Rolling:
        return Rolling(self, window)
