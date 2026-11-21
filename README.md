# vkdata

**Vulkan GPU-accelerated PyData, HoloViz, and scikit-learn — no CUDA required.**

`vkdata` is a unified Vulkan 1.3+ GPU data science, machine learning, and interactive visualization stack built on `llama.cpp`'s `ggml-vulkan` (plus `Tencent/ncnn` and direct SPIR-V compute shaders). Every package shares a **single `VkDevice` and zero-copy `VkBuffer`s**, running on AMD, Intel, NVIDIA, Apple (MoltenVK), and Steam Deck APUs.

---

## Packages

| Package | Mirrors | What It Provides |
|---|---|---|
| **`vknumpy`** | `NumPy` / `CuPy` | GPU `ndarray`, GEMM/BLAS, ufuncs, reductions, `argsort`, `top_k`, `groupby_agg`, `searchsorted`, `merge_indices`, zero-copy O(N) `bincount` / `histogram` / `histogram2d`, and C Vulkan buffer interop |
| **`vkpandas`** | `pandas` / `cuDF` | GPU `Series`, `DataFrame`, `read_parquet` / `read_csv`, filtering, `groupby`, `merge`, `sort_values`, `rolling`, `value_counts`, dictionary-encoded strings |
| **`vkshader`** | `Datashader` | Zero-copy GPU `Canvas`, `SpatialPoints`, GPU `eq_hist` / `log` / `cbrt` / `linear` shading, GPU `dynspread`, Fire colormap, VRAM-cached Esri World Imagery basemaps, and `anywidget` mouse/touch/box-zoom map |
| **`vkholoviews`** | `HoloViews` | Declarative `Points`, `Tiles`, `Histogram`, `Bars`, `Curve`, `*` Overlay, `+` Layout, `.opts()`, `datashade()` / `rasterize()`, and live GPU-linked cross-filtering |
| **`vklearn`** | `scikit-learn` / `cuML` | `StandardScaler`, `MinMaxScaler`, `LinearRegression`, `Ridge`, `LogisticRegression` (Newton-IRLS), `PCA`, `KMeans` (k-means++ + O(N) GPU `bincount`), `NearestNeighbors`, `KNeighborsClassifier`, `train_test_split`, `Pipeline`, and GPU metrics |

---

## Quick Start

```bash
# 1. Clone third_party (llama.cpp + ncnn) if needed
pixi run setup

# 2. Build both libvknumpy.so and libvkshader.so using all CPU cores
pixi run build

# 3. Run all 49 tests across all 5 packages
pixi run test

# 4. Launch JupyterLab with all example notebooks
pixi run lab
```

---

## Measured Benchmarks (AMD Steam Deck APU `RADV VANGOGH`, 11.84M Real NYC Taxi Trips)

- **Parquet → GPU `vkpandas` load (`nyc_taxi_wide.parq`):** `0.38s`
- **Zero-copy 2D point rasterization (`vknumpy.histogram2d`):** `5–10 ms`
- **GPU `eq_hist` + Fire LUT + bilinear Esri composite + `dynspread` (`vkshader`):** `1.75 ms`
- **Live linked 1D viewport histogram (`vknumpy.histogram` on 11.5M rows):** `1.2–7.0 ms`
- **End-to-end interactive Jupyter pan/zoom (including JPEG):** `15–24 ms` (~42–65 FPS)
- **GPU `Ridge` fare prediction (800k train / 200k test):** `0.13s` (`R² = 0.878`, `RMSE = $2.02`)
- **GPU `LogisticRegression` Newton-IRLS (800k train, 4 iters):** `0.25s`
- **GPU `KMeans` (`K=8` on 1,000,000 pickups, 20 iters):** `8.38s`

---

## Example Notebooks (`examples/`)
- [`examples/vkpandas_demo.ipynb`](examples/vkpandas_demo.ipynb) — Full `vkpandas` DataFrame walkthrough
- [`examples/nyc_taxi_vkshader.ipynb`](examples/nyc_taxi_vkshader.ipynb) — Terse 5-line interactive NYC Taxi map (`vkshader`)
- [`examples/nyc_taxi_holoviews.ipynb`](examples/nyc_taxi_holoviews.ipynb) — 4-panel Vulkan HoloViz dashboard + live linked cross-filtering (`vkholoviews`)
- [`examples/nyc_taxi_ml.ipynb`](examples/nyc_taxi_ml.ipynb) — GPU Fare & Tip (`$`) prediction, Tip classifier, K-Means hotspots, and interactive trip predictor (`vklearn`)
