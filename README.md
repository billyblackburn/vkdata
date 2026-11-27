# Hands-On AI, Machine Learning & GPU Computing — Without CUDA or ROCm

### *An Interactive, Follow-Along Book Powered by `vkdata` (Included!)*

> **Read on your phone or browser right now:** **[https://billyblackburn.github.io/vkdata/](https://billyblackburn.github.io/vkdata/)** &nbsp;|&nbsp; **[API Reference (17 Packages)](https://billyblackburn.github.io/vkdata/reference.html)** &nbsp;|&nbsp; **[Glossary](https://billyblackburn.github.io/vkdata/glossary.html)**

Most AI and machine learning books assume you own a modern NVIDIA CUDA GPU, a supported AMD ROCm card, or a cloud GPU subscription. **This book takes a different path:** it comes bundled with its own pure-Vulkan GPU toolkit (**`vkdata`**), so every single chapter runs live on the **Vulkan 1.3+ GPU you already own** — whether that's a **Steam Deck, AMD APU, older Radeon, Intel integrated/Arc GPU, older NVIDIA card, or Apple Silicon (MoltenVK)**.

You don't need to memorize a library manual first. Just open JupyterLab, start at Chapter 1, run each cell, tweak the numbers, and watch real AI models train on your own hardware.

---

## How to Follow Along

```bash
# 1. Clone third_party (llama.cpp + ncnn) and apply the Vulkan C accessor patch
pixi run setup

# 2. Build the native Vulkan backends (libvknumpy, libvkshader, libvkspatial)
pixi run build

# 3. Inspect your Vulkan GPU specs & run a 1-second compute/bandwidth check
pixi run gpu-specs

# 4. Verify everything works on your GPU (86 tests across all 17 included packages)
pixi run test

# 5. Open the interactive book in JupyterLab!
pixi run lab
```

> **Two ways to read this book:**
> - **Want results fast?** Start with **Part I (End-to-End Projects)** — go straight from messy real-world datasets to feature discovery, Classical ML, and Deep Learning.
> - **Want to understand every layer from the silicon up?** Start with **Part II (How GPUs Actually Think)** and work straight through to **Part VI (Modern LLMs)**.

---

## Table of Contents

### Part 0 · AI Foundations: Building Blocks of Neural Networks (`examples/00_foundations/`)
*Start from absolute zero: understand what a neuron, a hidden layer, an activation function, and "billions of parameters" actually mean.*

- **Foundations 1 — [The Perceptron: How a Single Artificial Neuron Works](examples/00_foundations/00a_the_perceptron.ipynb)**
  Weighted votes (`AND`, `OR`, `NOT`, `NAND`, `NOR`), watching a straight line learn with `vklearn.Perceptron`, running the same smooth neuron on Vulkan GPU (`vktorch.nn.Linear` + `Sigmoid`), and hitting the famous `XOR` limit.
- **Foundations 2 — [Multi-Layer Perceptrons (MLPs): Bending Space, Solving XOR & LLM Parameters](examples/00_foundations/00b_multi_layer_perceptron.ipynb)**
  Wire 3 perceptrons (`OR` + `NAND` → `AND`) to solve `XOR` by hand, prove why `GELU`/`ReLU` activations prevent linear collapse, watch 1 Neuron (`60%`) → Tiny MLP `2→4→1` (4-sided diamond, `100%`) → Deep MLP `2→32→32→1` (smooth circle, `100%`), and follow the parameter ladder from **3 knobs to 8 Billion in Llama 3**.

### Part I · The 5 Core AI Data Types (`examples/01_tabular/` → `05_graphs/`)
*Follow along across all 5 core AI data types — **Tabular**, **Image**, **Vector / Text**, **Time-Series**, and **Graph** — discovering features & benchmarking Classical ML first, then seeing if Deep Learning (`vktorch`) beats it on the exact same split.*

- **Chapter 1 *(Tabular Data)* — [California Housing: Feature Discovery & Classical ML](examples/01_tabular/01_housing_classical_ml.ipynb)**
  Load 20,640 Census districts, spot the `$500k` Census price cap on a GPU map, split without data leakage, engineer 36 domain + Haversine + `KMeans`-RBF features, and race `LinearRegression` → `Ridge` → GPU Distance-Weighted KNN (`$69.6k` → **`$52.7k` RMSE**, `0.640` → **`0.794` $R^2$**, plus an `87.5%` price-tier classifier).
- **Chapter 2 *(Tabular Data)* — [California Housing: Beating Classical ML with a Deep Neural Network](examples/01_tabular/02_housing_neural_network.ipynb)**
  Train a `vktorch` Pre-Norm Residual MLP (`LayerNorm` + `GELU` + `AdamW`) on the exact same split — comparing raw 8 features (`$50.3k`) vs. 36 engineered features (**`$48.2k` RMSE**, **`0.828` $R^2$** — beating GPU KNN by `-$4.5k` and the raw baseline by `-$21.4k`), plus `.npz` checkpoint saving & reloading.
- **Chapter 3 *(Image Data)* — [Fashion-MNIST: Image Feature Discovery & Classical Computer Vision](examples/02_images/03_fashion_classical_ml.ipynb)**
  Classify 10 clothing categories (`28 × 28` images): visualize "Average Clothing" prototypes, discover GPU **"Eigen-Clothes"** with `vklearn.PCA`, hand-craft Sobel/HOG edge features, and climb from `81.50%` (raw pixels) to **`86.45%`** — while inspecting the confusion matrix to see why `Shirt` stalls at `57.5%`.
- **Chapter 4 *(Image Data)* — [Fashion-MNIST: Beating Classical CV with a 2D CNN (`vktorch`)](examples/02_images/04_fashion_cnn_deep_learning.ipynb)**
  Train a modern strided `Conv2d` CNN (`15` epochs in **`24.5s`** on Steam Deck GPU with `12.8×` faster 100%-VRAM SPIR-V `im2col`/`col2im`!) directly on raw `28 × 28` images — beating every Classical CV model at **`87.70%`** (`Shirt` `+10.0 pts`), plus learned 3×3 filter & feature-map visuals.
- **Chapter 5 *(Vector / Text Data · Classical NLP)* — [Full-Length BBC News: TF-IDF Vectors, `vkvec` RAG Chunking & Classical NLP](examples/03_vectors_and_text/05_vector_search_classical_ml.ipynb)**
  Walk through the 3-step NLP pipeline on **2,225 complete multi-paragraph BBC News articles** (median 337 words, up to 4,492 words), chunk long articles into overlapping RAG passages with `vkvec.chunk_text`, run filtered Vulkan GPU Vector Search (`vkvec.VectorIndex`), visualize 5 topic islands with `vkvis.UMAP`, classify sections (`93.26%` BoW → **`97.30%`** TF-IDF + GPU Ridge), and hit the **Synonym Wall** (`0.000`).
- **Chapter 6 *(Vector / Text Data · Deep NLP & Hybrid Search)* — [Full-Length BBC News: Learned Neural Embeddings & `vkvec` Hybrid Search](examples/03_vectors_and_text/06_vector_embeddings_neural_net.ipynb)**
  Train a `vktorch` Neural Text Encoder (`18` epochs in **`12.2s`** on Steam Deck GPU) to compress full articles into dense **64-D meaning vectors** (`31×` smaller than TF-IDF), shatter the Synonym Wall (**`0.000` → `+0.994` / `+0.999`** on zero-word-overlap synonyms!), fuse **Okapi `BM25` + 64-D Neural Search** with `vkvec.HybridIndex` (`RRF`), and beat Classical NLP (**`97.53%`**) with `.npz` + `.vkvec` persistence.
- **Chapter 7 *(Time-Series Data · Astronomy & Classical ML)* — [277 Years of Solar Sunspots: Spectral Discovery & Classical Forecasting (`vktime`)](examples/04_time_series/07_solar_timeseries_classical_ml.ipynb)**
  Load 3,334 real Royal Observatory of Belgium solar observations (1749–2026), automatically discover the **11.11-year Solar Cycle** with GPU `vktime.autocorrelation` & `periodogram`, fold 25 cycles with `phase_fold`, and forecast 12 months ahead without lookahead leakage (`49.46` → **`35.39` RMSE**, `R² = 0.740`).
- **Chapter 8 *(Time-Series Data · Deep Temporal Neural Net)* — [Solar Cycle Forecasting: Beating Classical ML with `SolarForecasterNet` (`vktorch`)](examples/04_time_series/08_solar_timeseries_neural_net.ipynb)**
  Train a skip-connected Temporal Neural Network (`LayerNorm` + `GELU` + `AdamW` + `CosineAnnealingLR`) in **`14.1s`** on Steam Deck GPU — beating Classical Ridge across unseen 1973–2026 Solar Cycles 21–25 (**`34.56` RMSE**, **`R² = 0.752`**), forecasting late 2027, and saving/reloading `.npz` checkpoints.
- **Chapter 9 *(Graph Data · Networks & Classical ML)* — [Graph Data Explained: Karate Club, Cora Citations, GPU `PageRank` & `GraphRAG` (`vkgraph`)](examples/05_graphs/09_graph_classical_ml.ipynb)**
  Plain-English intro to Nodes, Edges, & Adjacency Matrices on Zachary's Karate Club (**`100%`** faction recovery from 2 leaders!), then scale to **2,708 real Cora AI papers** (`5,429` citations) with GPU `PageRank`, `BFS`, `GraphRAG` `k_hop_neighbors`, and GPU Label Propagation (`71.47%` words-only → **`85.50%`** citations / **`87.17%`** graph-smoothed).
- **Chapter 10 *(Graph Data · Graph Neural Network `GCN`)* — [Graph Deep Learning: Building a 2-Layer `GCN` in `vktorch`](examples/05_graphs/10_graph_neural_network_gcn.ipynb)**
  Add a single matrix multiply (`A_hat @ H @ W`) to turn a plain MLP (`73.87%`) into a **2-Layer Graph Convolutional Network** (**`87.35%`** in **`1.6s`** on Steam Deck GPU!), map 7 glowing AI topic islands with `vkvis.UMAP`, inspect the 7×7 confusion matrix, and persist `.npz` + `vkgraph` checkpoints.

---

### Part II · Under the Hood: How GPUs Actually Think (`examples/00_vulkan_gpu_puzzles/`)
*Write real GPU compute kernels in plain Python (`@vnp.kernel` → Python AST → GLSL 450 → SPIR-V → zero-copy Vulkan) before using higher-level ML libraries.*

- **Chapter 5 — [Puzzle 01: Map & the Python → GLSL → SPIR-V Pipeline](examples/00_vulkan_gpu_puzzles/puzzle_01_map.ipynb)** — Write your first GPU kernel with bounds guards, inspect the generated GLSL & SPIR-V, pass the auto-grader, and run a 1M-element benchmark.
- **Chapter 6 — [Puzzle 01b: Workgroup Sizes, Wavefronts & GPU Occupancy](examples/00_vulkan_gpu_puzzles/puzzle_01b_workgroup_sizes.ipynb)** — Sweep `local_size` from `1` to `1024` on 4M elements and see the 20× wavefront bandwidth cliff (`2.5 GB/s` → `50.9 GB/s`) with your own eyes.
- **Chapter 7 — [Puzzle 02: Zip, Memory Bandwidth & Fused AXPY](examples/00_vulkan_gpu_puzzles/puzzle_02_zip.ipynb)** — Multi-buffer GPU inputs (`out = a + b`), memory-bound vs. compute-bound intuition, and a fused `alpha * a + b` kernel (`0.30 ms` on 1M elements).

---

### Bonus Labs · Package Demos & 3D Deep Dives (`examples/bonus_labs/`)
*Standalone package demos, feature-engineering labs, and real-time 3D GPU showcases tucked neatly alongside the main book:*

- **Package Demos (`examples/bonus_labs/package_demos/`)**: [`vknumpy_demo.ipynb`](examples/bonus_labs/package_demos/vknumpy_demo.ipynb), [`vkpandas_demo.ipynb`](examples/bonus_labs/package_demos/vkpandas_demo.ipynb), [`vkscipy_demo.ipynb`](examples/bonus_labs/package_demos/vkscipy_demo.ipynb), [`nyc_taxi_holoviews.ipynb`](examples/bonus_labs/package_demos/nyc_taxi_holoviews.ipynb), [`nyc_taxi_vkshader.ipynb`](examples/bonus_labs/package_demos/nyc_taxi_vkshader.ipynb), [`vkcolorcet_demo.ipynb`](examples/bonus_labs/package_demos/vkcolorcet_demo.ipynb), [`vkplot_demo.ipynb`](examples/bonus_labs/package_demos/vkplot_demo.ipynb), [`vkpanel_demo.ipynb`](examples/bonus_labs/package_demos/vkpanel_demo.ipynb), [`feature_selection_and_engineering.ipynb`](examples/bonus_labs/package_demos/feature_selection_and_engineering.ipynb), [`vkvis_manifold_demo.ipynb`](examples/bonus_labs/package_demos/vkvis_manifold_demo.ipynb), [`vkspatial_demo.ipynb`](examples/bonus_labs/package_demos/vkspatial_demo.ipynb), [`vkgeoviews_demo.ipynb`](examples/bonus_labs/package_demos/vkgeoviews_demo.ipynb).
- **Deep Learning & 3D Showcases (`examples/bonus_labs/3d_and_deep_dives/`)**: [`backprop_and_regularization_lab.ipynb`](examples/bonus_labs/3d_and_deep_dives/backprop_and_regularization_lab.ipynb), [`vktorch_deep_learning_demo.ipynb`](examples/bonus_labs/3d_and_deep_dives/vktorch_deep_learning_demo.ipynb), [`autoencoder_and_mini_llama.ipynb`](examples/bonus_labs/3d_and_deep_dives/autoencoder_and_mini_llama.ipynb), [`nyc_taxi_ml.ipynb`](examples/bonus_labs/3d_and_deep_dives/nyc_taxi_ml.ipynb), [`sdss_cosmic_web_3d.ipynb`](examples/bonus_labs/3d_and_deep_dives/sdss_cosmic_web_3d.ipynb), [`earthquakes_3d_vkdata.ipynb`](examples/bonus_labs/3d_and_deep_dives/earthquakes_3d_vkdata.ipynb), [`opensky_vkdata.ipynb`](examples/bonus_labs/3d_and_deep_dives/opensky_vkdata.ipynb), [`vkastro_milky_way_3d.ipynb`](examples/bonus_labs/3d_and_deep_dives/vkastro_milky_way_3d.ipynb), [`protein_molecule_3d.ipynb`](examples/bonus_labs/3d_and_deep_dives/protein_molecule_3d.ipynb), [`nbody_galaxies_3d.ipynb`](examples/bonus_labs/3d_and_deep_dives/nbody_galaxies_3d.ipynb), [`ocean_fft_3d.ipynb`](examples/bonus_labs/3d_and_deep_dives/ocean_fft_3d.ipynb), [`wgpu_pygfx_nbody.ipynb`](examples/bonus_labs/3d_and_deep_dives/wgpu_pygfx_nbody.ipynb), [`datoviz_nbody_galaxies.ipynb`](examples/bonus_labs/3d_and_deep_dives/datoviz_nbody_galaxies.ipynb).

#### Book Sidebar: Why Fused GPU Kernels Make Deep Learning Faster (In Plain English)
A modern GPU can do math *ridiculously* fast — usually much faster than it can carry numbers back and forth from video memory (VRAM).

Think of it like cooking in a kitchen where the pantry (VRAM) is down the hall, and your cutting board (the GPU's tiny on-chip registers & shared memory) is right in front of you:
- **Unfused ops (the slow way):** To compute something like `LayerNorm` or `RMSNorm`, a naive framework runs 6–10 separate mini-programs in a row — compute the mean, walk back to the pantry and save it; fetch everything again to subtract the mean, walk back and save it; fetch it again to square it, and so on. Most of the time is spent walking back and forth to VRAM, not doing math!
- **Fused SPIR-V kernels (the fast way):** Instead of 10 separate trips, `vktorch` combines the entire forward or backward formula into **one single Vulkan compute shader** (`vktorch/_kernels.py`). Each group of GPU threads grabs a row of numbers once, does all the math right on the cutting board using fast shared memory, and writes only the final answer back to VRAM.

Here is the real measured speedup on a **Steam Deck APU** (`1024 × 256` tensor, full forward + backward pass):

| Operation | What It Does in an AI Model | Unfused (`vknumpy`) | Fused SPIR-V | Speedup |
|---|---|---:|---:|---:|
| **Conv2d (`im2col`/`col2im`)** | Rearranges 2D image patches for CNN convolutions in VRAM | 76.10 ms | **5.96 ms** | **12.8×** |
| **LayerNorm** | Keeps activations balanced in GPT-style Transformers | 19.98 ms | **6.09 ms** | **3.28×** |
| **RMSNorm** | Simpler, faster normalization used in Llama & Gemma | 15.34 ms | **5.87 ms** | **2.61×** |
| **SwiGLU** | Gated activation (`silu(gate) * up`) in modern LLM FFNs | 9.75 ms | **4.75 ms** | **2.05×** |
| **Softmax** | Turns raw attention scores into probabilities (adds to 1) | — | **4.64 ms** | Single-pass |
| **CrossEntropy** | Measures prediction error & computes gradients directly in VRAM | — | **1.73 ms** | No CPU one-hot |

*(This is the exact same core idea behind real-world AI accelerators like Triton, FlashAttention, and `torch.compile` — and you can read every line of our GLSL shaders in [`vktorch/_kernels.py`](vktorch/_kernels.py)!)*

---

## Appendix A · The Included `vkdata` Toolkit (17 Pure-Vulkan Packages)

Every chapter in this book runs on the bundled `vkdata` toolkit — powered by `llama.cpp`'s `ggml-vulkan`, `Tencent/ncnn`, and custom SPIR-V compute shaders sharing a **single `VkDevice` and zero-copy `VkBuffer`s**.

| Package | Familiar Equivalent | What It Does in This Book |
|---|---|---|
| **`vknumpy`** | `NumPy` / `CuPy` | GPU `ndarray`, `@vnp.kernel` Python-to-SPIR-V compiler, GEMM/BLAS, `einsum`, ufuncs, reductions, `linalg`, `argsort`, `top_k`, SPIR-V `bincount` / `histogram2d` |
| **`vkpandas`** | `pandas` / `cuDF` | GPU `DataFrame` / `Series`, `read_parquet` / `read_csv`, filtering, `groupby`, `merge`, `sort_values`, `rolling`, dictionary-encoded strings |
| **`vkscipy`** | `SciPy` | GPU `spatial.distance` (`cdist`/`pdist`), `stats` (`corrcoef`, `pearsonr`, `zscore`), `special`, `ndimage`, `signal`, `optimize` |
| **`vkshader`** | `Datashader` *(2D + 3D)* | Zero-copy SPIR-V 2D/3D point/line/sphere rasterizer, **0.91 ms SPIR-V 3-mip HDR Bloom + ACES**, Esri basemaps, & `anywidget` viewers |
| **`vkcolorcet`** | `Colorcet` | 26+ perceptually uniform colormaps cached as 256-entry GPU LUTs |
| **`vkholoviews`** | `HoloViews` | Declarative `Points`, `Path`, `Tiles`, `Histogram`, `Bars`, `Curve`, `*`/`+` composition, & live cross-filtering |
| **`vkgeoviews`** | `GeoViews` | CRS-aware geospatial `Points`, `Path`, `Polygons` choropleths & `tile_sources` |
| **`vkplot`** | `hvPlot` | High-level `.vkplot` EDA accessor on `DataFrame`, `Series`, and `GeoDataFrame` |
| **`vkpanel`** | `Panel` | Reactive `Row`, `Column`, `Card`, `Tabs`, `DataTable`, widgets & `pn.bind(...)` dashboards |
| **`vkspatial`** | `GeoPandas` / `cuSpatial` | Zero-copy SPIR-V 2D Morton `GeoDataFrame`, single-pass `sjoin`, GeoJSON I/O, Haversine, & **6-level 3D Barnes-Hut Octree (`BarnesHut3D`)** |
| **`vkvis`** | `mlx-vis` / `UMAP` | Pure-Vulkan GPU manifold learning: `UMAP`, `TSNE`, `PaCMAP`, `LocalMAP`, `TriMap`, `DREAMS`, `CNE`, `NNDescent`, & 2D/3D rendering |
| **`vklearn`** | `scikit-learn` / `cuML` | `StandardScaler`, `MinMaxScaler`, `LinearRegression`, `Ridge`, `LogisticRegression`, `PCA`, `KMeans`, `NearestNeighbors`, `KNeighborsClassifier`, `Pipeline` |
| **`vktorch`** | `PyTorch` | Dynamic reverse-mode autograd (`Tensor`, `.backward()`, `no_grad()`), `nn` (`Linear`, `LoRALinear`, `Conv2d`, `MaxPool2d`, `LayerNorm`, `RMSNorm`, `SwiGLU`, `Embedding`, `MultiheadAttention`, `KVCache`, `TransformerBlock`, `LlamaBlock`), fused SPIR-V kernels (incl. `RoPE` & `im2col`/`col2im`), `optim` (`SGD`, `Adam`, `AdamW`, `CosineAnnealingLR`), `DataLoader`, & `.npz` checkpointing |
| **`vkvec`** | `FAISS` / `Qdrant` | In-VRAM Vulkan GPU Vector Database (`VectorIndex` Cosine/L2/Dot + `where=` filtering), **Okapi `BM25Index`** full-text inverted index, **`HybridIndex` (Reciprocal Rank Fusion / RRF)**, GPU `TfidfVectorizer`, `cosine_similarity`, & RAG `chunk_text` |
| **`vktime`** | `Timescale` / `statsmodels` | In-VRAM Vulkan GPU Time-Series toolkit: `TimeSeries` (resampling, detrending, `.save()`/`.load()`), GPU rolling stats & `ewma`, `make_lag_matrix`, `sliding_windows`, leak-free `temporal_split`, GPU `autocorrelation`, `periodogram`, `seasonal_decompose`, robust `detect_anomalies`, & astronomy `phase_fold` |
| **`vkgraph`** | `NetworkX` / `PyG` / `cuGraph` | In-VRAM Vulkan GPU Graph Database & Analytics: `Graph` (edge lists or `vkpandas` DataFrames), GPU `pagerank`, `BFS` `shortest_path`, `GraphRAG` `k_hop_neighbors`, GPU `label_propagation`, `normalized_adjacency` for `vktorch` GCNs, `karate_club` / `load_cora`, & SynthWave `render_graph` |
| **`vkastro`** | `Astropy` | GPU `SkyCoord`, `aitoff_projection`, photometry, `FlatLambdaCDM` cosmology, & `read_fits` |

---

## Appendix B · Bonus Labs & 3D Showcases (`examples/bonus_labs/`)
See [`examples/README.md`](examples/README.md) for the complete book-matching index of all chapters, Vulkan GPU puzzles, and bonus labs.
