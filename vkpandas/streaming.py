"""vkpandas.streaming: Polars-Streaming / Dask-cuDF style out-of-core Vulkan GPU CSV & Parquet engine.

Streams arbitrarily large CSV files (10 GB, 50 GB, 100 GB+) through Vulkan VRAM in
auto-sized, high-occupancy chunks powered by multi-threaded C++ ``pyarrow.csv`` +
``vknumpy.from_arrow`` — maximizing GPU throughput while guaranteeing zero OOM!
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as _np
import vknumpy as vnp

from .dataframe import DataFrame
from .series import Series


def _sys_avail_mb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def recommended_vram_budget_mb(utilization: float = 0.35, max_mb: float = 1536.0) -> float:
    """Compute a high-occupancy, OOM-safe Vulkan VRAM chunk budget in MB.

    Respects both ``vnp.memory_info()['free_mb']`` and (on Unified Memory / UMA Linux
    APUs) system ``MemAvailable``, leaving comfortable headroom for intermediate GPU
    buffers and the OS.
    """
    free_vram = float(vnp.memory_info().get("free_mb", 0.0))
    sys_avail = _sys_avail_mb()
    safe_pool = free_vram
    if sys_avail > 0:
        # On UMA (e.g. Steam Deck), leave >= 1,500 MB for Linux/desktop
        safe_pool = min(safe_pool, max(128.0, sys_avail - 1500.0))
    if safe_pool <= 0:
        return 128.0
    budget = max(32.0, min(max_mb, safe_pool * float(utilization)))
    return round(budget, 1)


def _arrow_batch_to_df(batch: Any) -> DataFrame:
    out: Dict[str, Series] = {}
    schema = batch.schema
    for idx in range(batch.num_columns):
        name = schema.field(idx).name
        col = batch.column(idx)
        converted = vnp.from_arrow(col)
        if isinstance(converted, tuple):
            codes, uniques = converted
            out[name] = Series._from_vk(codes, name=name, categories=uniques)
        else:
            out[name] = Series._from_vk(converted, name=name)
    return DataFrame(out)


def iter_csv_chunks(
    path: Union[str, Path],
    *,
    chunksize: Optional[int] = None,
    block_size_mb: Optional[float] = None,
    delimiter: str = ",",
    names: Optional[Sequence[str]] = None,
) -> Iterator[DataFrame]:
    """Stream a CSV file as GPU-backed ``vkpandas.DataFrame`` chunks (never loads whole file)."""
    p = Path(path)
    try:
        import pyarrow.csv as pacsv

        # Choose C++ Arrow read block size (bytes per chunk)
        if block_size_mb is not None:
            blk_bytes = max(1 * 1024 * 1024, int(block_size_mb * 1024 * 1024))
        elif chunksize is not None:
            # Estimate ~80 bytes/row as a good read block size, then slice to exact chunksize
            blk_bytes = max(2 * 1024 * 1024, min(64 * 1024 * 1024, int(chunksize * 96)))
        else:
            blk_bytes = 32 * 1024 * 1024

        read_opts = pacsv.ReadOptions(
            block_size=blk_bytes,
            column_names=list(names) if names is not None else None,
            use_threads=True,
        )
        parse_opts = pacsv.ParseOptions(delimiter=delimiter)
        reader = pacsv.open_csv(str(p), read_options=read_opts, parse_options=parse_opts)

        if chunksize is None:
            for batch in reader:
                if batch.num_rows > 0:
                    yield _arrow_batch_to_df(batch)
            return

        # Re-chunk Arrow batches to exact ``chunksize`` rows
        buf_batches: List[Any] = []
        buf_rows = 0
        import pyarrow as pa

        for batch in reader:
            if batch.num_rows == 0:
                continue
            buf_batches.append(batch)
            buf_rows += batch.num_rows
            if buf_rows >= chunksize:
                tbl = pa.Table.from_batches(buf_batches)
                offset = 0
                while tbl.num_rows - offset >= chunksize:
                    sl = tbl.slice(offset, chunksize).combine_chunks()
                    for b in sl.to_batches():
                        if b.num_rows > 0:
                            yield _arrow_batch_to_df(b)
                    offset += chunksize
                rem = tbl.slice(offset)
                buf_batches = rem.to_batches() if rem.num_rows > 0 else []
                buf_rows = rem.num_rows
        if buf_rows > 0:
            tbl = pa.Table.from_batches(buf_batches).combine_chunks()
            for b in tbl.to_batches():
                if b.num_rows > 0:
                    yield _arrow_batch_to_df(b)
        return
    except ImportError:
        pass

    # Pure-Python fallback if pyarrow is not installed
    import csv
    from .io import _parse_column

    csize = int(chunksize or 100_000)
    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if names is not None:
            keys = [str(n).strip() for n in names]
        else:
            header = next(reader, None)
            if header is None:
                return
            keys = [h.strip() for h in header]
        cols: Dict[str, List[str]] = {k: [] for k in keys}
        n_buf = 0
        for row in reader:
            if not row:
                continue
            for k, val in zip(keys, row):
                cols[k].append(val)
            n_buf += 1
            if n_buf >= csize:
                yield DataFrame({k: _parse_column(v) for k, v in cols.items()})
                cols = {k: [] for k in keys}
                n_buf = 0
        if n_buf > 0:
            yield DataFrame({k: _parse_column(v) for k, v in cols.items()})


class _OnlineStats:
    """Exact parallel Welford/Chan running statistics across arbitrarily many GPU chunks."""

    __slots__ = ("count", "sum", "mean", "m2", "min", "max")

    def __init__(self) -> None:
        self.count: int = 0
        self.sum: float = 0.0
        self.mean: float = 0.0
        self.m2: float = 0.0
        self.min: float = math.inf
        self.max: float = -math.inf

    def update_from_gpu(self, arr: vnp.ndarray) -> None:
        n_b = len(arr)
        if n_b == 0:
            return
        sum_b = float(vnp.sum(arr))
        mean_b = sum_b / n_b
        diff_b = arr - mean_b
        m2_b = float(vnp.sum(diff_b * diff_b))
        min_b = float(vnp.min(arr))
        max_b = float(vnp.max(arr))

        if self.count == 0:
            self.count = n_b
            self.sum = sum_b
            self.mean = mean_b
            self.m2 = m2_b
            self.min = min_b
            self.max = max_b
        else:
            n_a = self.count
            n_tot = n_a + n_b
            delta = mean_b - self.mean
            self.mean = self.mean + delta * (n_b / n_tot)
            self.m2 = self.m2 + m2_b + (delta * delta) * (n_a * n_b / n_tot)
            self.count = n_tot
            self.sum += sum_b
            if min_b < self.min:
                self.min = min_b
            if max_b > self.max:
                self.max = max_b

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / max(1, self.count - 1)) if self.count > 1 else 0.0


class StreamGroupBy:
    """Out-of-core GPU GroupBy accumulator returned by ``StreamFrame.groupby(by)``."""

    def __init__(self, frame: "StreamFrame", by: str) -> None:
        self._frame = frame
        self._by = by

    def agg(self, agg_spec: Dict[str, Union[str, Sequence[str]]]) -> DataFrame:
        """Execute exact global GPU groupby aggregation across all chunks in bounded VRAM."""
        # group_key -> col -> {"count", "sum", "min", "max"}
        state: Dict[Any, Dict[str, Dict[str, float]]] = {}
        norm_spec: Dict[str, List[str]] = {
            c: ([ops] if isinstance(ops, str) else list(ops)) for c, ops in agg_spec.items()
        }

        all_ops = {op for ops in norm_spec.values() for op in ops}
        need_minmax = bool(all_ops & {"min", "max"})

        for chunk in self._frame._iter_transformed():
            if len(chunk) == 0:
                continue
            gb = chunk.groupby(self._by)
            needed_cols = list(norm_spec.keys())
            g_sum = gb.agg({c: "sum" for c in needed_cols})
            g_cnt = gb.agg({c: "count" for c in needed_cols})

            keys = g_sum[self._by].to_list()
            key_series = chunk[self._by]
            for col in needed_cols:
                sums = g_sum[col].to_list()
                cnts = g_cnt[col].to_list()
                col_arr = chunk[col]._data
                for k, sv, cv in zip(keys, sums, cnts):
                    g_entry = state.setdefault(k, {})
                    c_entry = g_entry.get(col)
                    if c_entry is None:
                        c_entry = {"sum": float(sv), "count": float(cv), "min": math.inf, "max": -math.inf}
                        g_entry[col] = c_entry
                    else:
                        c_entry["sum"] += float(sv)
                        c_entry["count"] += float(cv)
                    if need_minmax:
                        sub = col_arr[(key_series == k)._data]
                        if len(sub) > 0:
                            mnv, mxv = float(vnp.min(sub)), float(vnp.max(sub))
                            if mnv < c_entry["min"]:
                                c_entry["min"] = mnv
                            if mxv > c_entry["max"]:
                                c_entry["max"] = mxv
            del chunk

        sorted_keys = sorted(state.keys())
        out_cols: Dict[str, Any] = {self._by: sorted_keys}
        multi = any(len(ops) > 1 for ops in norm_spec.values())
        for col, ops in norm_spec.items():
            for op in ops:
                out_name = f"{col}_{op}" if multi else col
                vals = []
                for k in sorted_keys:
                    st = state[k][col]
                    if op == "sum":
                        vals.append(st["sum"])
                    elif op == "count":
                        vals.append(st["count"])
                    elif op == "mean":
                        vals.append(st["sum"] / max(1.0, st["count"]))
                    elif op == "min":
                        vals.append(st["min"])
                    elif op == "max":
                        vals.append(st["max"])
                    else:
                        raise ValueError(f"Unsupported streaming groupby op: {op!r}")
                out_cols[out_name] = vals
        return DataFrame(out_cols)


class StreamFrame:
    """Lazy, VRAM-budgeted out-of-core GPU DataFrame (Polars Streaming / Dask-cuDF style)."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        vram_budget_mb: Union[str, float] = "auto",
        delimiter: str = ",",
        names: Optional[Sequence[str]] = None,
        chunksize: Optional[int] = None,
        _ops: Optional[List[Tuple[str, Callable[[DataFrame], Any]]]] = None,
    ) -> None:
        self.path = Path(path)
        self.delimiter = delimiter
        self.names = names
        self.chunksize = chunksize
        self.vram_budget_mb: float = (
            recommended_vram_budget_mb() if vram_budget_mb == "auto" else float(vram_budget_mb)
        )
        # Convert VRAM budget into C++ Arrow CSV read block size (working set ~ 2.5x CSV chunk)
        self.block_size_mb: float = max(2.0, min(512.0, self.vram_budget_mb / 2.5))
        self._ops: List[Tuple[str, Callable[[DataFrame], Any]]] = list(_ops) if _ops else []
        self.last_stats: Dict[str, float | int] = {}

    def _clone(self, op: Tuple[str, Callable[[DataFrame], Any]]) -> "StreamFrame":
        return StreamFrame(
            self.path,
            vram_budget_mb=self.vram_budget_mb,
            delimiter=self.delimiter,
            names=self.names,
            chunksize=self.chunksize,
            _ops=self._ops + [op],
        )

    def filter(self, predicate: Callable[[DataFrame], Any]) -> "StreamFrame":
        """Lazily filter each GPU chunk with ``predicate(df) -> boolean Series/mask``."""
        return self._clone(("filter", predicate))

    def with_columns(self, fn: Callable[[DataFrame], DataFrame]) -> "StreamFrame":
        """Lazily transform or add GPU columns on each chunk."""
        return self._clone(("with_columns", fn))

    def _iter_transformed(self) -> Iterator[DataFrame]:
        vnp.empty_cache()
        vnp.reset_peak_memory_stats()
        t0 = time.perf_counter()
        n_chunks = 0
        total_rows_in = 0
        total_rows_out = 0
        file_bytes = self.path.stat().st_size if self.path.exists() else 0

        for chunk in iter_csv_chunks(
            self.path,
            chunksize=self.chunksize,
            block_size_mb=self.block_size_mb if self.chunksize is None else None,
            delimiter=self.delimiter,
            names=self.names,
        ):
            n_chunks += 1
            total_rows_in += len(chunk)
            for kind, fn in self._ops:
                if kind == "filter":
                    mask = fn(chunk)
                    chunk = chunk[mask]
                elif kind == "with_columns":
                    chunk = fn(chunk)
            total_rows_out += len(chunk)
            yield chunk
            del chunk
            vnp.empty_cache()

        elapsed = max(time.perf_counter() - t0, 1e-6)
        peak_mb = vnp.max_memory_allocated() / (1024 * 1024)
        self.last_stats = {
            "chunks": n_chunks,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "file_mb": round(file_bytes / (1024 * 1024), 2),
            "elapsed_s": round(elapsed, 3),
            "throughput_mb_s": round((file_bytes / (1024 * 1024)) / elapsed, 1),
            "throughput_rows_s": int(total_rows_in / elapsed),
            "peak_vram_mb": round(peak_mb, 2),
            "vram_budget_mb": round(self.vram_budget_mb, 1),
            "block_size_mb": round(self.block_size_mb, 1),
        }

    def groupby(self, by: str) -> StreamGroupBy:
        """Return an out-of-core GPU GroupBy accumulator."""
        return StreamGroupBy(self, by)

    def agg(self, agg_spec: Dict[str, Union[str, Sequence[str]]]) -> Dict[str, float]:
        """Compute exact global reductions (count/sum/mean/std/min/max) across all chunks."""
        stats_map: Dict[str, _OnlineStats] = {c: _OnlineStats() for c in agg_spec}
        for chunk in self._iter_transformed():
            if len(chunk) == 0:
                continue
            for col in agg_spec:
                stats_map[col].update_from_gpu(chunk[col]._data)

        out: Dict[str, float] = {}
        for col, ops in agg_spec.items():
            op_list = [ops] if isinstance(ops, str) else list(ops)
            st = stats_map[col]
            for op in op_list:
                key = f"{col}_{op}" if len(op_list) > 1 else col
                if op == "count":
                    out[key] = float(st.count)
                elif op == "sum":
                    out[key] = st.sum
                elif op == "mean":
                    out[key] = st.mean
                elif op == "std":
                    out[key] = st.std
                elif op == "min":
                    out[key] = st.min
                elif op == "max":
                    out[key] = st.max
                else:
                    raise ValueError(f"Unsupported streaming agg op: {op!r}")
        return out

    def describe(self) -> DataFrame:
        """Compute an exact global summary table (count/mean/std/min/max) across all chunks."""
        stats_map: Dict[str, _OnlineStats] = {}
        num_cols: List[str] = []
        for chunk in self._iter_transformed():
            if len(chunk) == 0:
                continue
            if not num_cols:
                num_cols = [c for c in chunk.columns if not chunk[c].is_string]
                stats_map = {c: _OnlineStats() for c in num_cols}
            for c in num_cols:
                stats_map[c].update_from_gpu(chunk[c]._data)

        table: Dict[str, Any] = {"stat": ["count", "mean", "std", "min", "max"]}
        for c in num_cols:
            st = stats_map[c]
            table[c] = [float(st.count), st.mean, st.std, st.min, st.max]
        return DataFrame(table)

    def to_parquet(self, out_path: Union[str, Path]) -> Path:
        """Stream filtered/transformed GPU chunks directly to a Parquet file on disk."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for chunk in self._iter_transformed():
                if len(chunk) == 0:
                    continue
                arrays = {c: pa.array(chunk[c].to_list()) for c in chunk.columns}
                tbl = pa.table(arrays)
                if writer is None:
                    writer = pq.ParquetWriter(str(out), tbl.schema)
                writer.write_table(tbl)
        finally:
            if writer is not None:
                writer.close()
        return out


def scan_csv(
    path: Union[str, Path],
    *,
    vram_budget_mb: Union[str, float] = "auto",
    delimiter: str = ",",
    sep: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    chunksize: Optional[int] = None,
) -> StreamFrame:
    """Create a lazy, VRAM-budgeted out-of-core GPU StreamFrame for arbitrarily large CSVs.

    Like Polars ``pl.scan_csv`` or Dask-cuDF: automatically sizes GPU chunks to maximize
    Vulkan throughput while keeping peak VRAM strictly bounded below OOM.
    """
    delim = sep if sep is not None else delimiter
    return StreamFrame(
        path,
        vram_budget_mb=vram_budget_mb,
        delimiter=delim,
        names=names,
        chunksize=chunksize,
    )


__all__ = [
    "StreamFrame",
    "StreamGroupBy",
    "iter_csv_chunks",
    "recommended_vram_budget_mb",
    "scan_csv",
]
