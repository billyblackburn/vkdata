# Hands-On AI, Machine Learning & GPU Computing — Without CUDA or ROCm

### *An Interactive, Follow-Along Book Powered by `vkdata` (Included!)*

> **Read on your phone or browser right now:** **[https://billyblackburn.github.io/vkdata/](https://billyblackburn.github.io/vkdata/)**

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

# 4. Verify everything works on your GPU (72 tests across all 14 included packages)
pixi run test

# 5. Open the interactive book in JupyterLab!
pixi run lab
```

> **Two ways to read this book:**
> - **Want results fast?** Start with **Part I (End-to-End Projects)** — go straight from messy real-world datasets to feature discovery, Classical ML, and Deep Learning.
> - **Want to understand every layer from the silicon up?** Start with **Part II (How GPUs Actually Think)** and work straight through to **Part VI (Modern LLMs)**.

---

## Table of Contents

### Part 0 · AI Foundations: Building Blocks of Neural Networks (`examples/00_ai_foundations/`)
*Start from absolute zero: understand what a neuron, a hidden layer, an activation function, and "billions of parameters" actually mean.*

- **Foundations 1 — [The Perceptron: How a Single Artificial Neuron Works](examples/00_ai_foundations/01_the_perceptron.ipynb)**
  Weighted votes (`AND`, `OR`, `NOT`, `NAND`, `NOR`), watching a straight line learn with `vklearn.Perceptron`, running the same smooth neuron on Vulkan GPU (`vktorch.nn.Linear` + `Sigmoid`), and hitting the famous `XOR` limit.
- **Foundations 2 — [Multi-Layer Perceptrons (MLPs): Bending Space, Solving XOR & LLM Parameters](examples/00_ai_foundations/02_multi_layer_perceptron.ipynb)**
  Wire 3 perceptrons (`OR` + `NAND` → `AND`) to solve `XOR` by hand, prove why `GELU`/`ReLU` activations prevent linear collapse, watch 1 Neuron (`60%`) → Tiny MLP `2→4→1` (4-sided diamond, `100%`) → Deep MLP `2→32→32→1` (smooth circle, `100%`), and follow the parameter ladder from **3 knobs to 8 Billion in Llama 3**.

### Part I · End-to-End Projects: Tabular, Image & Vector Data (`examples/tldr_end2end/`)
*Follow along across the 3 core AI data types — **Tabular**, **Image**, and **Vector / Text** — discovering features & benchmarking Classical ML first, then seeing if Deep Learning (`vktorch`) beats it on the exact same split.*

- **Chapter 1 *(Tabular Data)* — [California Housing: Feature Discovery & Classical ML](examples/tldr_end2end/01_housing_classical_ml.ipynb)**
  Load 20,640 Census districts, spot the `$500k` Census price cap on a GPU map, split without data leakage, engineer 36 domain + Haversine + `KMeans`-RBF features, and race `LinearRegression` → `Ridge` → GPU Distance-Weighted KNN (`$69.6k` → **`$52.7k` RMSE**, `0.640` → **`0.794` $R^2$**, plus an `87.5%` price-tier classifier).
- **Chapter 2 *(Tabular Data)* — [California Housing: Beating Classical ML with a Deep Neural Network](examples/tldr_end2end/02_housing_neural_network.ipynb)**
  Train a `vktorch` Pre-Norm Residual MLP (`LayerNorm` + `GELU` + `AdamW`) on the exact same split — comparing raw 8 features (`$50.3k`) vs. 36 engineered features (**`$48.2k` RMSE**, **`0.828` $R^2$** — beating GPU KNN by `-$4.5k` and the raw baseline by `-$21.4k`), plus `.npz` checkpoint saving & reloading.
- **Chapter 3 *(Image Data)* — [Fashion-MNIST: Image Feature Discovery & Classical Computer Vision](examples/tldr_end2end/03_fashion_classical_ml.ipynb)**
  Classify 10 clothing categories (`28 × 28` images): visualize "Average Clothing" prototypes, discover GPU **"Eigen-Clothes"** with `vklearn.PCA`, hand-craft Sobel/HOG edge features, and climb from `81.50%` (raw pixels) to **`86.45%`** — while inspecting the confusion matrix to see why `Shirt` stalls at `57.5%`.
- **Chapter 4 *(Image Data)* — [Fashion-MNIST: Beating Classical CV with a 2D CNN (`vktorch`)](examples/tldr_end2end/04_fashion_cnn_deep_learning.ipynb)**
  Train a modern strided `Conv2d` CNN (`15` epochs in **`24.5s`** on Steam Deck GPU with `12.8×` faster 100%-VRAM SPIR-V `im2col`/`col2im`!) directly on raw `28 × 28` images — beating every Classical CV model at **`87.70%`** (`Shirt` `+10.0 pts`), plus learned 3×3 filter & feature-map visuals.
- **Chapter 5 *(Vector / Text Data)* — [AG News: TF-IDF Vectors, GPU Vector Search & Classical ML](examples/tldr_end2end/05_vector_search_classical_ml.ipynb)**
  Turn 7,600 real news articles into L2-normalized 2,000-D vectors, build a **5-line Vulkan GPU Vector Database** (`~2.6 ms` Cosine Similarity + `top_k`), visualize 4 topic islands with `vkvis.UMAP`, classify topics (`82.12%` BoW → **`86.44%`** TF-IDF + GPU Ridge), and hit the classic **Synonym Wall** (`0.000` similarity on `"car"` vs. `"automobile"`).

---

### Part II · Under the Hood: How GPUs Actually Think (`examples/00_vulkan_gpu_puzzles/`)
*Write real GPU compute kernels in plain Python (`@vnp.kernel` → Python AST → GLSL 450 → SPIR-V → zero-copy Vulkan) before using higher-level ML libraries.*

- **Chapter 5 — [Puzzle 01: Map & the Python → GLSL → SPIR-V Pipeline](examples/00_vulkan_gpu_puzzles/puzzle_01_map.ipynb)** — Write your first GPU kernel with bounds guards, inspect the generated GLSL & SPIR-V, pass the auto-grader, and run a 1M-element benchmark.
- **Chapter 6 — [Puzzle 01b: Workgroup Sizes, Wavefronts & GPU Occupancy](examples/00_vulkan_gpu_puzzles/puzzle_01b_workgroup_sizes.ipynb)** — Sweep `local_size` from `1` to `1024` on 4M elements and see the 20× wavefront bandwidth cliff (`2.5 GB/s` → `50.9 GB/s`) with your own eyes.
- **Chapter 7 — [Puzzle 02: Zip, Memory Bandwidth & Fused AXPY](examples/00_vulkan_gpu_puzzles/puzzle_02_zip.ipynb)** — Multi-buffer GPU inputs (`out = a + b`), memory-bound vs. compute-bound intuition, and a fused `alpha * a + b` kernel (`0.30 ms` on 1M elements).

---

### Part III · GPU Data Foundations & Visual Exploration (`examples/01_gpu_foundations/` & `02_eda_and_visualization/`)
*Manipulate arrays, tables, and millions of data points interactively on your GPU before modeling.*

- **Chapter 8 — [GPU Arrays & Linear Algebra (`vknumpy`)](examples/01_gpu_foundations/vknumpy_demo.ipynb)** — 1024×1024 GPU matrix multiplication vs. CPU, `einsum`, linear solvers, & 5M-point SPIR-V histograms.
- **Chapter 9 — [GPU DataFrames (`vkpandas`)](examples/01_gpu_foundations/vkpandas_demo.ipynb)** — Filtering, `groupby`, `merge`, and Parquet I/O directly in VRAM.
- **Chapter 10 — [GPU Scientific Computing (`vkscipy`)](examples/01_gpu_foundations/vkscipy_demo.ipynb)** — Pairwise distances, Sobel/Gaussian image filters, spectrograms, & nonlinear curve fitting.
- **Chapter 11 — [Visual EDA on Millions of Rows](examples/02_eda_and_visualization/nyc_taxi_holoviews.ipynb)** — Interactive 11.84M-point NYC Taxi explorer, plus focused labs on [perceptual colormaps (`vkcolorcet`)](examples/02_eda_and_visualization/vkcolorcet_demo.ipynb), [`.vkplot`](examples/02_eda_and_visualization/vkplot_demo.ipynb), [reactive dashboards (`vkpanel`)](examples/02_eda_and_visualization/vkpanel_demo.ipynb), and [5-line GPU datashading (`vkshader`)](examples/02_eda_and_visualization/nyc_taxi_vkshader.ipynb).

---

### Part IV · Feature Engineering, Selection & Manifold Learning (`examples/03_feature_engineering_and_selection/`)
*Turn messy raw columns into clean, high-signal features — and visualize high-dimensional data.*

- **Chapter 12 — [Feature Selection & Engineering Lab](examples/03_feature_engineering_and_selection/feature_selection_and_engineering.ipynb)** — Variance threshold, GPU collinearity pruning (`vkscipy.stats.corrcoef`), engineered interactions, permutation importance, & before/after neural network ablation.
- **Chapter 13 — [Manifold Learning & Dimensionality Reduction (`vkvis`)](examples/03_feature_engineering_and_selection/vkvis_manifold_demo.ipynb)** — 7-algorithm GPU comparison (`UMAP`, `TSNE`, `PaCMAP`, `LocalMAP`, `TriMap`, `DREAMS`, `CNE`), 3D Swiss Roll unrolling, and animated UMAP training.
- **Chapter 14 — [Geospatial Feature Engineering (`vkspatial` & `vkgeoviews`)](examples/03_feature_engineering_and_selection/vkspatial_demo.ipynb)** — Haversine distances, 1M-point Morton spatial indexes, `point_in_polygon`, and [single-pass GPU spatial joins on 11.84M NYC trips](examples/03_feature_engineering_and_selection/vkgeoviews_demo.ipynb).

---

### Part V · Classical Machine Learning on Real Datasets (`examples/04_classical_ml/`)
*Supervised regression, classification, and unsupervised clustering at scale.*

- **Chapter 15 — [End-to-End Tabular ML: NYC Taxi Fares & Tips](examples/04_classical_ml/nyc_taxi_ml.ipynb)** — Fare & tip regression, tip classification, K-Means hotspots, & interactive trip predictor.
- **Chapter 16 — [Unsupervised 3D Clustering: 150,000 SDSS Galaxies](examples/04_classical_ml/sdss_cosmic_web_3d.ipynb)** — 3D GPU K-Means on the real SDSS DR17 cosmic web + Color–Magnitude Diagram.
- **Chapter 17 — [Geospatial Hotspots: 2.12 Million Real Earthquakes](examples/04_classical_ml/earthquakes_3d_vkdata.ipynb)** — Spatial joins, GPU K-Means clustering, & 3D subduction-zone explorer on USGS seismic data.

---

### Part VI · Deep Learning, Backpropagation & Modern LLMs (`examples/05_deep_learning/`)
*Learn how autograd, regularization, CNNs, Autoencoders, and Llama-style Transformers work under the hood with `vktorch`.*

- **Chapter 18 — [Backpropagation & Regularization Visual Lab](examples/05_deep_learning/backprop_and_regularization_lab.ipynb)** — Step-by-step chain-rule gradients & side-by-side overfitting vs. `Dropout` + `AdamW` weight decay.
- **Chapter 19 — [MLPs, 2D CNNs & a Causal Mini-GPT Transformer](examples/05_deep_learning/vktorch_deep_learning_demo.ipynb)** — Train a 3-Spiral MLP, a 2D `Conv2d` CNN, and a Mini-GPT live on your Vulkan GPU.
- **Chapter 20 — [Denoising Autoencoders & Modern Mini-Llama (`RMSNorm` + `SwiGLU`)](examples/05_deep_learning/autoencoder_and_mini_llama.ipynb)** — Visualize an Autoencoder's latent space with `vkvis.UMAP` and train a Mini-Llama that generates coherent sentences in seconds.

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

## Appendix A · The Included `vkdata` Toolkit (15 Pure-Vulkan Packages)

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
| **`vkvec`** | `FAISS` / `ChromaDB` | In-VRAM Vulkan GPU Vector Database (`VectorIndex` with Cosine/L2/Dot search, `where=` metadata filtering, `.save()`/`.load()`), GPU `TfidfVectorizer`, `cosine_similarity`, & RAG `chunk_text` |
| **`vkastro`** | `Astropy` | GPU `SkyCoord`, `aitoff_projection`, photometry, `FlatLambdaCDM` cosmology, & `read_fits` |

---

## Appendix B · Bonus GPU Physics & Scientific Showcases (`examples/extras_and_showcase/`)
*Extra real-time GPU simulations & 3D visualizations to explore after finishing the main book:*
- [`opensky_vkdata.ipynb`](examples/extras_and_showcase/opensky_vkdata.ipynb) — 10.23M real OpenSky flight trajectories (2D + 3D Europe orbit)
- [`vkastro_milky_way_3d.ipynb`](examples/extras_and_showcase/vkastro_milky_way_3d.ipynb) — 400,000 real ESA Gaia DR3 stars & Hertzsprung–Russell diagram
- [`protein_molecule_3d.ipynb`](examples/extras_and_showcase/protein_molecule_3d.ipynb) — Real RCSB PDB proteins (`6VXX` 23.7k-atom Spike Trimer) as 3D Vulkan sphere impostors
- [`nbody_galaxies_3d.ipynb`](examples/extras_and_showcase/nbody_galaxies_3d.ipynb) — 16,384-body 3D Barnes-Hut colliding galaxies + 0.91 ms SPIR-V HDR Bloom (~41 FPS)
- [`ocean_fft_3d.ipynb`](examples/extras_and_showcase/ocean_fft_3d.ipynb) — Vulkan GPU Tessendorf 2D IFFT ocean simulation
- [`wgpu_pygfx_nbody.ipynb`](examples/extras_and_showcase/wgpu_pygfx_nbody.ipynb) — WebGPU (`wgpu` + `pygfx`) 16,384-body colliding galaxies (~96 FPS)
- [`datoviz_nbody_galaxies.ipynb`](examples/extras_and_showcase/datoviz_nbody_galaxies.ipynb) & [`datoviz_nbody_galaxies.py`](examples/extras_and_showcase/datoviz_nbody_galaxies.py) — Native Vulkan 60+ FPS Datoviz window with Dear ImGui HUD
