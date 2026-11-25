# vkdata

**Learn how AI actually works on the GPU you already own — no ROCm, no modern CUDA GPU, no cloud bill required.**

Most GPU machine learning and deep learning tutorials assume you own a modern NVIDIA CUDA card or a supported AMD ROCm datacenter/desktop GPU. **`vkdata`** opens up the **entire AI & ML lifecycle** on **any Vulkan 1.3+ GPU** — including **Steam Deck & AMD APUs, older Radeon cards without ROCm, Intel integrated/Arc GPUs, older NVIDIA cards, and Apple Silicon (MoltenVK)**.

Experiment hands-on with your own hardware from Stage 1 to Stage 5: GPU arrays & tabular dataframes → visual EDA on millions of rows → feature engineering & feature selection → dimensionality reduction → classical ML → reverse-mode autograd, CNNs, & causal Transformers — all powered by `llama.cpp`'s `ggml-vulkan`, `Tencent/ncnn`, and custom SPIR-V compute shaders sharing a **single `VkDevice` and zero-copy `VkBuffer`s**.

### Who Is This For?
- **Learners & tinkerers without ROCm or a modern CUDA GPU** who want to run real GPU-accelerated ML and deep learning on their laptop, mini-PC, or handheld.
- **Curious developers** who want to see *how* AI works under the hood — feature selection, PCA/UMAP, gradient descent, backpropagation, convolutions, and self-attention — on transparent, readable Python + Vulkan code.

---

## Learning Path (`examples/`)

Follow the numbered AI & ML curriculum from Stage `00` → Stage `05`, or jump straight into a real-world **TL;DR End-to-End** project:

### TL;DR End-to-End Projects (`examples/tldr_end2end/`)
*Start with messy real-world datasets, discover & engineer features on GPU, and compare Classical ML vs. Neural Networks on the exact same split.*
- [`01_housing_classical_ml.ipynb`](examples/tldr_end2end/01_housing_classical_ml.ipynb) — **California Housing (Feature Discovery & Classical ML):** load 20,640 Census districts with `vkpandas`, GPU price map & `$500k` cap EDA, leak-free split & imputation, engineer 36 domain/spatial/`KMeans`-RBF features, & bake-off `LinearRegression` → `Ridge` → GPU Distance-Weighted KNN (`$69.6k` → **`$52.7k` RMSE**, `0.640` → **`0.794` $R^2$**, plus `87.5%` price-tier classifier)
- [`02_housing_neural_network.ipynb`](examples/tldr_end2end/02_housing_neural_network.ipynb) — **California Housing (Deep Learning with `vktorch`):** Pre-Norm Residual MLP (`LayerNorm` + `GELU` + `AdamW` fused SPIR-V kernels) on the exact same split — Raw 8-feature ResNet (`$50.3k`) vs. Engineered 36-feature ResNet (**`$48.2k` RMSE**, **`0.828` $R^2$**, beating GPU KNN by `-$4.5k` & raw baseline by `-$21.4k`), plus `.npz` checkpoint roundtrip

### Stage 00 · Vulkan GPU Puzzles (`examples/00_vulkan_gpu_puzzles/`)
*Write real GPU compute kernels in Python syntax (`@vnp.kernel` → Python AST → GLSL 450 → SPIR-V → zero-copy Vulkan).*
- [`puzzle_01_map.ipynb`](examples/00_vulkan_gpu_puzzles/puzzle_01_map.ipynb) — Explains the Python AST → GLSL → SPIR-V pipeline + **Puzzle 01 (`Map`)** with bounds guards, live AST/GLSL inspection, auto-grader, & 1M-element GPU benchmark
- [`puzzle_01b_workgroup_sizes.ipynb`](examples/00_vulkan_gpu_puzzles/puzzle_01b_workgroup_sizes.ipynb) — **Workgroup Sizes, Wavefronts & GPU Occupancy**: sweeps `local_size` from `1` to `1024` on 4M elements, visually showing the 20× wavefront bandwidth cliff (`2.5 GB/s` → `50.9 GB/s`)
- [`puzzle_02_zip.ipynb`](examples/00_vulkan_gpu_puzzles/puzzle_02_zip.ipynb) — **Puzzle 02 (`Zip`)**: multi-buffer Vulkan SSBOs (`out = a + b`), memory-bandwidth bound vs. compute-bound intuition, & **Fused AXPY** (`alpha * a + b`) with a `float` push constant (`0.30 ms` on 1M elements)

### Stage 01 · GPU Foundations (`examples/01_gpu_foundations/`)
*Arrays, linear algebra, dataframes, and scientific primitives on Vulkan GPU.*
- [`vknumpy_demo.ipynb`](examples/01_gpu_foundations/vknumpy_demo.ipynb) — 1024×1024 GPU GEMM vs CPU, ufuncs, `einsum`, `linalg.solve`, 5M SPIR-V `bincount` / `histogram` / `histogram2d`
- [`vkpandas_demo.ipynb`](examples/01_gpu_foundations/vkpandas_demo.ipynb) — GPU `DataFrame` / `Series`, filtering, `groupby`, `merge`, Parquet I/O
- [`vkscipy_demo.ipynb`](examples/01_gpu_foundations/vkscipy_demo.ipynb) — 2048×2048 GPU `cdist`, `ndimage` Gaussian/Sobel, GPU GEMM chirp `spectrogram`, 200k-point `curve_fit`

### Stage 02 · Exploratory Data Analysis & Visualization (`examples/02_eda_and_visualization/`)
*Inspect millions of rows interactively before modeling.*
- [`vkcolorcet_demo.ipynb`](examples/02_eda_and_visualization/vkcolorcet_demo.ipynb) — Perceptually uniform colormaps & 26 GPU LUT palettes on 2M points
- [`vkplot_demo.ipynb`](examples/02_eda_and_visualization/vkplot_demo.ipynb) — `hvPlot`-style `.vkplot` accessor across 11.84M NYC Taxi + 10.23M OpenSky flights (2D + 3D)
- [`vkpanel_demo.ipynb`](examples/02_eda_and_visualization/vkpanel_demo.ipynb) — Reactive `pn.bind`, `Tabs`, `Card`, `DataTable`, sliders & dropdowns on 2M GPU points
- [`nyc_taxi_vkshader.ipynb`](examples/02_eda_and_visualization/nyc_taxi_vkshader.ipynb) — Terse 5-line interactive 11.84M-point NYC Taxi GPU datashader map
- [`nyc_taxi_holoviews.ipynb`](examples/02_eda_and_visualization/nyc_taxi_holoviews.ipynb) — 4-panel HoloViz dashboard + live cross-filtered map + neighborhood polygon & landmark radius explorer

### Stage 03 · Feature Engineering, Selection & Manifold Learning (`examples/03_feature_engineering_and_selection/`)
*Clean, select, engineer, and project high-dimensional features on GPU.*
- [`feature_selection_and_engineering.ipynb`](examples/03_feature_engineering_and_selection/feature_selection_and_engineering.ipynb) — **Complete feature selection workflow**: GPU variance threshold, `vkscipy.stats.corrcoef` collinearity pruning, engineered interaction features, GPU permutation importance, & before/after `vktorch` MLP ablation
- [`vkvis_manifold_demo.ipynb`](examples/03_feature_engineering_and_selection/vkvis_manifold_demo.ipynb) — `mlx-vis`-style GPU dimensionality reduction: 7-algorithm comparison (`UMAP`, `TSNE`, `PaCMAP`, `LocalMAP`, `TriMap`, `DREAMS`, `CNE`), 3D Swiss Roll unrolling, live UMAP trajectory player, & 3D sphere orbit
- [`vkspatial_demo.ipynb`](examples/03_feature_engineering_and_selection/vkspatial_demo.ipynb) — Geospatial feature engineering: WGS84 ↔ Web Mercator, Haversine, 1M 2D Morton `GeoDataFrame`, `radius_query` & `point_in_polygon`
- [`vkgeoviews_demo.ipynb`](examples/03_feature_engineering_and_selection/vkgeoviews_demo.ipynb) — Single-pass GPU `sjoin()` zone-level feature aggregation on 11.84M NYC trips → dark-canvas choropleth & Esri satellite overlay

### Stage 04 · Classical Machine Learning (`examples/04_classical_ml/`)
*Supervised regression/classification & unsupervised clustering on real datasets.*
- [`nyc_taxi_ml.ipynb`](examples/04_classical_ml/nyc_taxi_ml.ipynb) — End-to-end tabular ML: GPU Fare & Card-Tip (`$`) regression, Tip classifier, K-Means hotspots, & interactive trip predictor (`vklearn`)
- [`sdss_cosmic_web_3d.ipynb`](examples/04_classical_ml/sdss_cosmic_web_3d.ipynb) — Unsupervised 3D GPU K-Means clustering on 150,000 real SDSS DR17 galaxies + Color–Magnitude Diagram
- [`earthquakes_3d_vkdata.ipynb`](examples/04_classical_ml/earthquakes_3d_vkdata.ipynb) — Spatial join + GPU K-Means hotspot discovery + 3D subduction-zone explorer on 2.12M real USGS earthquakes

### Stage 05 · Deep Learning & Transformers (`examples/05_deep_learning/`)
*Reverse-mode autograd, MLPs, CNNs, Autoencoders, and modern LLMs on Vulkan GPU (powered by fused SPIR-V kernels for `AdamW`, `SGD`, `GELU`, `SiLU`, `SwiGLU`, `RMSNorm`, `LayerNorm`, `Softmax`, and `CrossEntropy`).*
- [`backprop_and_regularization_lab.ipynb`](examples/05_deep_learning/backprop_and_regularization_lab.ipynb) — Visual lab: step-by-step chain-rule backpropagation gradients & overfitting vs. `Dropout` + `AdamW` weight decay decision boundaries
- [`vktorch_deep_learning_demo.ipynb`](examples/05_deep_learning/vktorch_deep_learning_demo.ipynb) — Train a **3-Spiral MLP** (100% acc + 2D decision boundary), a **2D `Conv2d` CNN** (100% acc), and a **Causal Mini-GPT Transformer** live on Vulkan GPU
- [`autoencoder_and_mini_llama.ipynb`](examples/05_deep_learning/autoencoder_and_mini_llama.ipynb) — **Denoising Autoencoder** latent manifold visualized via `vkvis.UMAP` + **Modern Mini-Llama (`RMSNorm` + `SwiGLU` + Causal Attention)** text generation

#### Why Fused GPU Kernels Make Deep Learning Faster (In Plain English)
A modern GPU can do math *ridiculously* fast — usually much faster than it can carry numbers back and forth from video memory (VRAM).

Think of it like cooking in a kitchen where the pantry (VRAM) is down the hall, and your cutting board (the GPU's tiny on-chip registers & shared memory) is right in front of you:
- **Unfused ops (the slow way):** To compute something like `LayerNorm` or `RMSNorm`, a naive framework runs 6–10 separate mini-programs in a row — compute the mean, walk back to the pantry and save it; fetch everything again to subtract the mean, walk back and save it; fetch it again to square it, and so on. Most of the time is spent walking back and forth to VRAM, not doing math!
- **Fused SPIR-V kernels (the fast way):** Instead of 10 separate trips, `vktorch` combines the entire forward or backward formula into **one single Vulkan compute shader** (`vktorch/_kernels.py`). Each group of GPU threads grabs a row of numbers once, does all the math right on the cutting board using fast shared memory, and writes only the final answer back to VRAM.

Here is the real measured speedup on a **Steam Deck APU** (`1024 × 256` tensor, full forward + backward pass):

| Operation | What It Does in an AI Model | Unfused (`vknumpy`) | Fused SPIR-V | Speedup |
|---|---|---:|---:|---:|
| **LayerNorm** | Keeps activations balanced in GPT-style Transformers | 19.98 ms | **6.09 ms** | **3.28×** |
| **RMSNorm** | Simpler, faster normalization used in Llama & Gemma | 15.34 ms | **5.87 ms** | **2.61×** |
| **SwiGLU** | Gated activation (`silu(gate) * up`) in modern LLM FFNs | 9.75 ms | **4.75 ms** | **2.05×** |
| **Softmax** | Turns raw attention scores into probabilities (adds to 1) | — | **4.64 ms** | Single-pass |
| **CrossEntropy** | Measures prediction error & computes gradients directly in VRAM | — | **1.73 ms** | No CPU one-hot |

*(This is the exact same core idea behind real-world AI accelerators like Triton, FlashAttention, and `torch.compile` — and you can read every line of our GLSL shaders in [`vktorch/_kernels.py`](vktorch/_kernels.py)!)*

### Extras & GPU Showcase (`examples/extras_and_showcase/`)
*Additional real-time GPU physics simulations & scientific visualization demos outside the core ML curriculum.*
- [`opensky_vkdata.ipynb`](examples/extras_and_showcase/opensky_vkdata.ipynb) — 10.23M real OpenSky flight trajectories: 2D density/velocity/country maps & full-Europe 3D orbit
- [`vkastro_milky_way_3d.ipynb`](examples/extras_and_showcase/vkastro_milky_way_3d.ipynb) — 400,000 real ESA Gaia DR3 stars: Hertzsprung–Russell diagram, all-sky Galactic Aitoff map, 3D Solar Neighborhood, & `FlatLambdaCDM` cosmology
- [`protein_molecule_3d.ipynb`](examples/extras_and_showcase/protein_molecule_3d.ipynb) — Real RCSB PDB proteins (`6VXX` 23.7k-atom SARS-CoV-2 Spike Trimer, `6M0J`, `4HHB`) as Vulkan 3D Blinn–Phong sphere impostors + GPU `vkscipy.cdist` contact maps
- [`nbody_galaxies_3d.ipynb`](examples/extras_and_showcase/nbody_galaxies_3d.ipynb) — 16,384-body 6-level 3D Barnes-Hut Morton octree (`vkspatial.BarnesHut3D`, ~1.2 ms/step) + 0.91 ms native SPIR-V 3-mip HDR Bloom + ACES tone mapping (~41 FPS)
- [`ocean_fft_3d.ipynb`](examples/extras_and_showcase/ocean_fft_3d.ipynb) — Vulkan GPU recreation of NVIDIA's CUDA `oceanFFT` (256×256 Tessendorf 2D IFFT + auto-animating 3D wave mesh)
- [`wgpu_pygfx_nbody.ipynb`](examples/extras_and_showcase/wgpu_pygfx_nbody.ipynb) — Zero-copy WebGPU (`wgpu` + `pygfx`) 16,384-body Barnes-Hut colliding galaxies + 98,304 trails + `PhysicalBasedBloomPass` (~96 FPS)
- [`datoviz_nbody_galaxies.ipynb`](examples/extras_and_showcase/datoviz_nbody_galaxies.ipynb) & [`datoviz_nbody_galaxies.py`](examples/extras_and_showcase/datoviz_nbody_galaxies.py) — Native Vulkan hardware-rasterized 60+ FPS Datoviz window with Barred Spiral × Cartwheel Ring / 3-Way Merger presets, 3-layer HDR bloom halos, & Dear ImGui HUD

---

## Packages (14 Included)

| Package | Mirrors | Role in the ML / DL Lifecycle |
|---|---|---|
| **`vknumpy`** | `NumPy` / `CuPy` | GPU `ndarray`, GEMM/BLAS, `einsum`, ufuncs, reductions, `linalg`, `argsort`, `top_k`, `groupby_agg`, `searchsorted`, SPIR-V `bincount` / `histogram2d` |
| **`vkpandas`** | `pandas` / `cuDF` | GPU `DataFrame` / `Series`, `read_parquet` / `read_csv`, filtering, `groupby`, `merge`, `sort_values`, `rolling`, dictionary-encoded strings |
| **`vkscipy`** | `SciPy` | GPU `spatial.distance` (`cdist`/`pdist`), `stats` (`corrcoef`, `pearsonr`, `zscore`), `special` (`softmax`, `expit`), `ndimage`, `signal`, `optimize` |
| **`vkshader`** | `Datashader` *(2D + 3D)* | Zero-copy SPIR-V 2D/3D point/line/sphere rasterizer, **0.91 ms SPIR-V 3-mip HDR Bloom + ACES (`hdr_bloom`)**, Esri basemaps, & `anywidget` 2D/3D viewers |
| **`vkcolorcet`** | `Colorcet` | 26+ perceptually uniform colormaps cached as 256-entry GPU LUTs + `swatches()` |
| **`vkholoviews`** | `HoloViews` | Declarative `Points`, `Path`, `Tiles`, `Histogram`, `Bars`, `Curve`, `*`/`+` composition, & live cross-filtering `.interact()` |
| **`vkgeoviews`** | `GeoViews` | CRS-aware geospatial `Points`, `Path`, `Polygons` choropleths & `tile_sources` |
| **`vkplot`** | `hvPlot` | High-level `.vkplot` EDA accessor on `DataFrame`, `Series`, and `GeoDataFrame` |
| **`vkpanel`** | `Panel` | Reactive `Row`, `Column`, `Card`, `Tabs`, `DataTable`, widgets & `pn.bind(...)` dashboards |
| **`vkspatial`** | `GeoPandas` / `cuSpatial` | Zero-copy SPIR-V 2D Morton `GeoDataFrame`, single-pass `sjoin`, GeoJSON I/O, `to_crs`, & **6-level 3D Barnes-Hut Octree (`BarnesHut3D`)** |
| **`vkvis`** | `mlx-vis` / `UMAP` | Pure-Vulkan GPU dimensionality reduction & manifold learning: `UMAP`, `TSNE`, `PaCMAP`, `LocalMAP`, `TriMap`, `DREAMS`, `CNE`, `NNDescent`, & 2D/3D rendering |
| **`vklearn`** | `scikit-learn` / `cuML` | `StandardScaler`, `MinMaxScaler`, `LinearRegression`, `Ridge`, `LogisticRegression`, `PCA`, `KMeans`, `NearestNeighbors`, `KNeighborsClassifier`, `Pipeline` |
| **`vktorch`** | `PyTorch` | Dynamic reverse-mode autograd (`Tensor`, `.backward()`, `no_grad()`), `nn` (`Linear`, `Conv2d`, `MaxPool2d`, `LayerNorm`, `RMSNorm`, `SwiGLU`, `BatchNorm1d`, `Embedding`, `MultiheadAttention`, `TransformerBlock`, `LlamaBlock`), fused SPIR-V kernels, `optim` (`SGD`, `Adam`, `AdamW`, `CosineAnnealingLR`), `DataLoader`, & `.npz` checkpointing |
| **`vkastro`** | `Astropy` | GPU `SkyCoord` (`ICRS` ↔ `Galactic`, 3D Cartesian, `cone_search`), `aitoff_projection`, photometry, `FlatLambdaCDM` cosmology, & `read_fits` |

---

## Quick Start

```bash
# 1. Clone third_party (llama.cpp + ncnn) and apply Vulkan C accessor patch
pixi run setup

# 2. Build libvknumpy.so, libvkshader.so, and libvkspatial.so using all CPU cores
pixi run build

# 3. Run all 71 tests across all 14 packages
pixi run test

# 4. Build & auditwheel-repair a manylinux wheel (for PyPI / pip install)
pixi run wheel

# 5. Launch JupyterLab with the 5-stage AI & ML curriculum
pixi run lab
```
