# `vkpandas` Readiness Roadmap (Separate Track)

This document tracks the GPU primitives and columnar data features needed in `vknumpy` to support building a Vulkan GPU DataFrame library (**`vkpandas`** — `Series`, `DataFrame`, filtering, `groupby`, `merge`/`join`, `sort_values`, `rolling`, `value_counts`) on top of it.

It is intentionally kept **separate from the core NumPy drop-in milestones** in [`docs/ROADMAP.md`](ROADMAP.md).

---

## Already Available in `vknumpy` Today
- GPU-resident numeric columns (`float32`, `float16`, `int32` device buffers)
- Elementwise arithmetic, comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`), `where`, logicals (`and`/`or`/`xor`/`not`)
- Column aggregations: `sum`, `mean`, `std`, `var`, `max`, `min`, `prod`, `argmax`, `argmin`, `count_nonzero`
- Row gather (`take` via `ggml_get_rows`), sorting (`argsort`, `sort`, `top_k`), prefix sums (`cumsum`), `concatenate`, `roll` (`shift`), `pad`

---

## Required Additions for `vkpandas`

### Track A — Native `int32` / `bool_` GPU Ops (Keys, Indices & Validity Masks) *(Done)*
- Native GPU `int32` gather via `ggml_get_rows` (`get_rows_i32`), GPU `float32 ↔ int32` cast (`vkn_cast`), and GPU `int32` comparisons, `argsort`, and `cumsum`.
- Required for row indices, group IDs, timestamps, join keys, categorical codes, and null/validity masks.

### Track B — Stream Compaction & Scatter (`df[df["x"] > 0]`, `dropna`) *(Done)*
- GPU stream compaction (`flatnonzero`, `nonzero`, `compress`) built purely from `ggml-vulkan` (`ggml_cumsum` + `ggml_arange` + `ggml_set_rows` + `ggml_cpy` — zero custom shaders).
- General GPU scatter (`scatter_rows`, `put` via `ggml_set_rows`), plus GPU boolean mask indexing (`a[mask]`) and fancy integer indexing (`a[idx]`) on `ndarray.__getitem__`.

### Track C — Segmented Reductions & `bincount` (`groupby`, `unique`, `value_counts`, `drop_duplicates`)
- Sort-based `groupby` pipeline on GPU: `argsort(key)` → `take(key, order)` → boundary detection (`sorted[1:] != sorted[:-1]`) → `cumsum` group IDs → segmented aggregation.
- Primitives needed: `unique`, `bincount`, and indexed/segmented reductions (`scatter_add` / `add.at`, `maximum.at`, `minimum.at`, `reduceat`).

### Track D — Parallel Binary Search (`merge` / `join`, `isin`, `cut`)
- `searchsorted(sorted_a, v, side="left"|"right")` on GPU for sort-merge joins (`inner`/`left`/`outer`), `merge_asof`, and `pd.cut`/`histogram`.
- `isin(a, test_elements)` built from `sort` + `searchsorted`.

### Track E — Missing-Value / `NaN` Semantics (`skipna=True`, `isna`, `fillna`)
- GPU `isnan`, `isinf`, `isfinite`, `nan_to_num`.
- Masked `nan*` reductions (`nansum`, `nanmean`, `nanstd`, `nanmax`, `nanmin`) and `ffill`/`bfill`.

### Track F — Multi-Key Sorting, Rolling Windows, Strings & Columnar I/O
- Stable multi-column `lexsort` for `df.sort_values(["a", "b"])`.
- Rolling window ops (`rolling(w).sum()/mean()/max()`) via GPU `cumsum` and `ggml_pool_1d` / `ggml_conv_1d`.
- Dictionary-encoded string/categorical columns (`int32` codes on GPU + host string dictionary) so string `==`, `groupby`, `value_counts`, and `merge` run at full GPU `int32` speed.
- Zero-copy `pyarrow` / Parquet / CSV columnar ingestion directly into `vknumpy` device buffers.

---

## Candidate Open-Source Vulkan Libraries to Evaluate (Before Writing Custom Shaders)
- **Google `vk-radix-sort`** / **AMD FidelityFX Parallel Sort** (MIT) for high-throughput `int32`/`uint32` GPU radix sort & key-value sorting.
- **Existing `ggml-vulkan` kernels** already compiled into `libvknumpy.so`: `ggml_set_rows` (scatter), `ggml_pool_1d` / `ggml_conv_1d` (rolling windows).
