# vkdata

**Vulkan GPU-accelerated PyData, SciPy, HoloViz, GeoSpatial, Astrophysics, and scikit-learn — no CUDA required.**

`vkdata` is a unified Vulkan 1.3+ GPU scientific computing, machine learning, geospatial, astrophysics, and interactive 2D/3D visualization stack built on `llama.cpp`'s `ggml-vulkan` (plus `Tencent/ncnn` and direct SPIR-V compute shaders). Every package shares a **single `VkDevice` and zero-copy `VkBuffer`s**, running on AMD, Intel, NVIDIA, Apple (MoltenVK), and Steam Deck APUs.

---

## Packages (12 Included)

| Package | Mirrors | What It Provides |
|---|---|---|
| **`vknumpy`** | `NumPy` / `CuPy` | GPU `ndarray`, GEMM/BLAS, `einsum`, ufuncs, reductions, `linalg`, `argsort`, `top_k`, `groupby_agg`, `searchsorted`, `merge_indices`, zero-copy O(N) SPIR-V `bincount` / `histogram` / `histogram2d`, and C Vulkan buffer interop |
| **`vkpandas`** | `pandas` / `cuDF` | GPU `Series`, `DataFrame`, `read_parquet` (with column projection) / `read_csv`, filtering, `groupby`, `merge`, `sort_values`, `rolling`, `value_counts`, dictionary-encoded strings |
| **`vkscipy`** | `SciPy` | GPU `spatial.distance` (`cdist` / `pdist` / `squareform` via GEMM), `stats` (`norm`, `zscore`, `pearsonr`, `corrcoef`, `skew`, `kurtosis`), `special` (`expit`, `erf`, `ndtr`, `softmax`, `logsumexp`), `ndimage` (`gaussian_filter`, `sobel`, `laplace`, `convolve`), `signal` (`convolve`, `correlate`, batched GEMM `spectrogram`), `optimize` (`curve_fit`, `minimize`) |
| **`vkspatial`** | `GeoPandas` / `SpatialPandas` / `cuSpatial` | Zero-copy SPIR-V 2D Morton Z-order `GeoDataFrame`, `point_in_polygon`, single-pass multi-polygon `sjoin` (`sjoin_polygons.comp`), zero-dependency GeoJSON I/O (`read_file`/`read_geojson`/`to_geojson`), `to_crs` (`EPSG:4326` ↔ `EPSG:3857`), `cx` slicer, `total_bounds`, `bbox`, `radius_query`, `nearest`, `haversine_distance`, `bearing`, `point_to_segment_distance`, `polygon_area`/`centroid` |
| **`vkshader`** | `Datashader` *(2D + 3D)* | Zero-copy SPIR-V 2D `Canvas.points` & `Canvas.line` (`count`, `mean`, categorical RGB), GPU `eq_hist`/`log`/`cbrt`/`linear` shading with `vkcolorcet` LUTs, GPU `dynspread`, VRAM-cached multi-provider tiles (`EsriImagery`, `EsriStreet`, `EsriDark`/`Light`, `OSM`), **`anywidget` 2D pan/zoom/box-zoom**, and **real-time 3D perspective volumetric renderer (`render_3d` / `interact_3d`)** with tilted satellite ground plane, orbit, zoom, & Shift/Right-drag panning |
| **`vkcolorcet`** | `Colorcet` | 26+ perceptually uniform sequential, diverging, cyclic, and categorical colormaps (`fire`, `bmy`, `bgy`, `kbc`, `viridis`, `inferno`, `plasma`, `magma`, `turbo`, `coolwarm`, `rainbow`, `glasbey`, ...) cached as 256-entry GPU LUTs + `swatches()` viewer |
| **`vkholoviews`** | `HoloViews` | Declarative `Points`, `Path`, `Tiles`, `Histogram`, `Bars`, `Curve`, `*` Overlay, `+` Layout, `.opts()`, `datashade()` / `rasterize()`, and live GPU-linked cross-filtering `.interact()` (with live `Shade:`, `Cmap:`, and `Spread:` controls) |
| **`vkgeoviews`** | `GeoViews` | CRS-aware `Points`, `Path`, and `Polygons` choropleths with automatic GPU `EPSG:4326` → `EPSG:3857` projection and `tile_sources` (`EsriImagery`, `EsriStreet`, `CartoDark`, `CartoLight`, `OSM`) |
| **`vkplot`** | `hvPlot` | High-level `.vkplot` accessor on `vkpandas.DataFrame`, `Series`, and `vkspatial.GeoDataFrame` (`.points`, `.scatter`, `.paths`, `.hist`, `.bar`, `.line`, `.plot3d`) |
| **`vkpanel`** | `Panel` | Reactive `Row`, `Column`, `Card`, `Tabs`, `Pane`, `DataTable`, sliders/dropdowns/checkboxes/buttons, and `pn.bind(...)` dashboards |
| **`vklearn`** | `scikit-learn` / `cuML` | `StandardScaler`, `MinMaxScaler`, `LinearRegression`, `Ridge`, `LogisticRegression` (Newton-IRLS), `PCA`, `KMeans` (k-means++ + O(N) GPU `bincount`), `NearestNeighbors`, `KNeighborsClassifier`, `train_test_split`, `Pipeline`, and GPU metrics |
| **`vkastro`** | `Astropy` | GPU `SkyCoord` (`ICRS` ↔ `Galactic` Hipparcos 3×3 rotation, 3D Cartesian parsecs, spherical `separation`, `cone_search`), `aitoff_projection`, photometry (`parallax_to_distance`, `absolute_magnitude`, `luminosity_solar`, `bp_rp_to_temperature`), `FlatLambdaCDM` cosmology, and zero-dependency `read_fits` |

---

## Quick Start

```bash
# 1. Clone third_party (llama.cpp + ncnn) and apply Vulkan C accessor patch
pixi run setup

# 2. Build libvknumpy.so, libvkshader.so, and libvkspatial.so using all CPU cores
pixi run build

# 3. Run all 62 tests across all 12 packages
pixi run test

# 4. Build & auditwheel-repair a manylinux wheel (for PyPI / pip install)
pixi run wheel

# 5. Launch JupyterLab with all 15 example notebooks
pixi run lab
```

---

## Measured Benchmarks (AMD Steam Deck APU `RADV VANGOGH`)

- **NYC Taxi (11.84M rows):**
  - Parquet → GPU `vkpandas` load: **`0.38 s`**
  - Zero-copy 2D point rasterization (`vknumpy.histogram2d`): **`5–10 ms`**
  - GPU `eq_hist` + LUT + bilinear Esri composite + `dynspread` (`vkshader`): **`1.75 ms`**
  - Live linked 1D viewport histogram (`vknumpy.histogram` on 11.5M rows): **`1.2–7.0 ms`**
  - End-to-end interactive Jupyter pan/zoom (including JPEG): **`15–24 ms`** (~42–65 FPS)
  - Zero-copy GPU `vkspatial.morton_encode` (11.84M pts): **`3.73 ms`** · `point_in_polygon`: **`38.5 ms`**
  - GPU `Ridge` fare prediction (800k train): **`0.13 s`** (`R² = 0.878`, `RMSE = $2.02`) · `LogisticRegression` Newton-IRLS: **`0.25 s`**
- **OpenSky Flights (10.23M trajectory points):**
  - Zero-copy GPU `Canvas.line` + `eq_hist` Fire shade: **`40.4 ms`**
  - Interactive 3D Vulkan GPU orbit (`vkshader.interact_3d`, London + tilted Esri satellite): **`46.1 ms/frame`** (~22 FPS) · Full Europe 3D: **`88.6 ms/frame`**
- **ESA Gaia DR3 (400,000 real stars):**
  - Load + `ICRS → Galactic` + 3D Cartesian + Aitoff + photometry on GPU: **`< 0.5 s`**

---

## Example Notebooks (`examples/`)

### Structural Biology & Molecular (`examples/biology/`)
- [`protein_molecule_3d.ipynb`](examples/biology/protein_molecule_3d.ipynb) — Real RCSB PDB proteins (`6VXX` 23.7k-atom SARS-CoV-2 Spike Trimer, `6M0J` Spike+ACE2, `4HHB` Hemoglobin) rendered as Vulkan 3D Blinn–Phong shaded sphere impostors + GPU `vkscipy.cdist` protein contact map

### Astronomy & Astrophysics (`examples/astronomy/`)
- [`vkastro_milky_way_3d.ipynb`](examples/astronomy/vkastro_milky_way_3d.ipynb) — 400,000 real ESA Gaia DR3 stars: Hertzsprung–Russell diagram, all-sky Galactic Aitoff map, 3D Solar Neighborhood fly-through, Pleiades cone search, & `FlatLambdaCDM` cosmology
- [`sdss_cosmic_web_3d.ipynb`](examples/astronomy/sdss_cosmic_web_3d.ipynb) — 150,000 real SDSS DR17 galaxies: Color–Magnitude Diagram (Red Sequence vs. Blue Cloud), 2D/3D Cosmic Web in Megaparsecs, & GPU K-Means 3D cluster regions

### Geoscience & Geospatial (`examples/geoscience/`)
- [`earthquakes_3d_vkdata.ipynb`](examples/geoscience/earthquakes_3d_vkdata.ipynb) — 2.12M real USGS earthquakes: global seismic map, tectonic `sjoin` choropleth, 3D subduction-zone explorer (depth up to 650 km), & GPU K-Means hotspots
- [`vkgeoviews_demo.ipynb`](examples/geoscience/vkgeoviews_demo.ipynb) — Single-pass GPU `sjoin()` on 11.84M NYC trips → dark-canvas Mean-Fare Choropleth & Esri satellite overlay
- [`vkspatial_demo.ipynb`](examples/geoscience/vkspatial_demo.ipynb) — WGS84 ↔ Web Mercator, Haversine, 1M 2D Morton `GeoDataFrame`, `radius_query` & `point_in_polygon`

### Transportation & Mobility (`examples/transportation/`)
- [`opensky_vkdata.ipynb`](examples/transportation/opensky_vkdata.ipynb) — 10.23M OpenSky flight trajectories: density, country of origin, ascending/descending, velocity, 6-city airport close-ups, & full-Europe 3D orbit
- [`nyc_taxi_holoviews.ipynb`](examples/transportation/nyc_taxi_holoviews.ipynb) — 4-panel HoloViz dashboard + live cross-filtered map + NYC neighborhood polygon & landmark radius explorer
- [`nyc_taxi_vkshader.ipynb`](examples/transportation/nyc_taxi_vkshader.ipynb) — Terse 5-line interactive 11.84M-point NYC Taxi map

### Machine Learning (`examples/machine_learning/`)
- [`nyc_taxi_ml.ipynb`](examples/machine_learning/nyc_taxi_ml.ipynb) — GPU Fare & Card-Tip (`$`) prediction, Tip classifier, K-Means hotspots, and interactive trip predictor (`vklearn`)

### Core Compute & Visualization (`examples/core_and_viz/`)
- [`vknumpy_demo.ipynb`](examples/core_and_viz/vknumpy_demo.ipynb) — 1024×1024 GPU GEMM vs CPU, ufuncs, `einsum`, `linalg.solve`, 5M SPIR-V `bincount` / `histogram` / `histogram2d`
- [`vkpandas_demo.ipynb`](examples/core_and_viz/vkpandas_demo.ipynb) — GPU `DataFrame` / `Series`, filtering, `groupby`, `merge`, Parquet I/O
- [`vkscipy_demo.ipynb`](examples/core_and_viz/vkscipy_demo.ipynb) — 2048×2048 GPU `cdist`, `ndimage` Gaussian/Sobel, GPU GEMM chirp `spectrogram`, 200k-point `curve_fit`
- [`vkcolorcet_demo.ipynb`](examples/core_and_viz/vkcolorcet_demo.ipynb) — Interactive swatch catalog + all 26 primary colormaps rendered on 2M GPU points
- [`vkpanel_demo.ipynb`](examples/core_and_viz/vkpanel_demo.ipynb) — Reactive `pn.bind`, `Tabs`, `Card`, `DataTable`, sliders & dropdowns on 2M GPU points
- [`vkplot_demo.ipynb`](examples/core_and_viz/vkplot_demo.ipynb) — `hvPlot`-style `.vkplot` accessor across 11.84M NYC Taxi + 10.23M OpenSky flights (2D + 3D)
