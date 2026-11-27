# `vkdata` Book & Vulkan GPU Toolkit Roadmap

This roadmap tracks what is complete and what's next for both the **interactive follow-along book** ([billyblackburn.github.io/vkdata](https://billyblackburn.github.io/vkdata/)) and the bundled **pure-Vulkan GPU toolkit** (`vkdata`).

---

## Completed Milestones

### Interactive Book & Learning Platform
- **Online SynthWave '84 Book (`docs/`):** Mobile-friendly HTML chapters with Python syntax highlighting, sticky Prev/Next navigation, **66-term searchable Glossary** ([`docs/glossary.html`](docs/glossary.html)), and **in-chapter tap-to-explain acronym popovers** (`250+` links across prose & inline code).
- **Part 0 · AI Foundations (`examples/00_ai_foundations/`):**
  - *The Perceptron:* All 5 basic single-neuron logic gates (`AND`, `OR`, `NOT`, `NAND`, `NOR`), `vklearn.Perceptron`, Vulkan GPU smooth neuron, & the `XOR` wall.
  - *Multi-Layer Perceptrons (MLPs):* Solving `XOR`, why `GELU`/`ReLU` activations prevent linear collapse, 1 Neuron vs. Tiny MLP vs. Deep MLP on concentric circles, & the 3-to-8B LLM parameter ladder.
- **Part I · End-to-End Projects (`examples/tldr_end2end/`):**
  - *Ch. 1–2 · Tabular Data (California Housing):* Classical ML (`$69.6k` → `$52.7k` GPU KNN) vs. `vktorch` Residual MLP (**`$48.2k` RMSE**, `R² = 0.828`).
  - *Ch. 3–4 · Image Data (Fashion-MNIST):* Classical CV (`81.50%` → `86.45%` PCA + Edge/HOG) vs. `vktorch` 2D CNN (**`87.70%`**, `Shirt` `+10.0 pts`, **24.5s** on Steam Deck APU).
  - *Ch. 5–6 · Vector / Text Data & NLP (Full-Length BBC News):* `vkvec` TF-IDF + RAG chunking + GPU Vector DB (`97.30%`) vs. `vktorch` 64-D Neural Embeddings (**`97.53%`**, Synonym Wall broken `0.000 → +0.999`, plus **Hybrid BM25 + Neural RRF Search**).
  - *Ch. 7–8 · Time-Series & Astronomy (SILSO Solar Sunspots, 1749–2026):* `vktime` 11.1-yr spectral discovery + Classical Ridge (`35.39` RMSE) vs. `vktorch` `SolarForecasterNet` (**`34.56` RMSE**, `R² = 0.752`, `14.1s` on Steam Deck GPU).
  - *Ch. 9–10 · Graph & Network Data (Karate Club + Cora AI Citations):* `vkgraph` GPU `PageRank`, `BFS`, `GraphRAG`, & Label Propagation (`85.50%`) vs. `vktorch` 2-Layer `GCN` (**`87.35%`** in `1.6s` on Steam Deck GPU, `+13.5 pts` over plain MLP).
- **Parts II–VI Notebooks (`examples/`):** Vulkan GPU Puzzles (`@vnp.kernel`), GPU Foundations, Visual EDA on 11.84M rows, Feature Selection & 7-algorithm UMAP/Manifold Learning, Classical ML at scale, Backprop Lab, Autoencoders, & Mini-Llama (`RMSNorm` + `SwiGLU`).
- **Hardware Inspector:** [`scripts/gpu_specs.py`](scripts/gpu_specs.py) (`pixi run gpu-specs`) for instant Vulkan device limits + 1-second FP32 GEMM & SPIR-V bandwidth check.

### Bundled Pure-Vulkan Toolkit (16 Packages · 82/82 Tests Passing)
- **Core Compute, Data & Search:** `vknumpy` (`ggml-vulkan` + `ncnn` + `@vnp.kernel` Python-to-SPIR-V compiler), `vkpandas` (full GPU DataFrame, filtering, `groupby`, `merge`, Parquet/CSV), `vkscipy`, `vkspatial` (Morton 2D `GeoDataFrame`, `sjoin`, 6-level 3D Barnes-Hut Octree), **`vkvec`** (GPU `VectorIndex`, `BM25Index`, `HybridIndex` with Reciprocal Rank Fusion, `TfidfVectorizer`, `chunk_text`), **`vktime`** (GPU `TimeSeries`, rolling windows, lag matrices, `temporal_split`, `autocorrelation`, `periodogram`, `detect_anomalies`, `phase_fold`), `vkastro`.
- **Visualization & Dashboards:** `vkshader` (2D/3D SPIR-V rasterizer + 0.91 ms HDR Bloom), `vkcolorcet`, `vkholoviews` (multi-layer `Points * Curve` overlays & layouts), `vkgeoviews`, `vkplot`, `vkpanel`, `vkvis` (UMAP/t-SNE/PaCMAP/TriMap/CNE).
- **Machine Learning & Deep Learning:** `vklearn` (`LinearRegression`, `Ridge`, `LogisticRegression`, `Perceptron`, `PCA`, `KMeans`, `NearestNeighbors`, `KNeighborsClassifier`, `Pipeline`) and `vktorch` (reverse-mode autograd, `Conv2d`, `TransformerBlock`, `LlamaBlock`, `.npz` checkpointing, and **fused single-pass SPIR-V kernels** for `AdamW`, `SGD`, `GELU`, `SiLU`, `SwiGLU`, `RMSNorm`, `LayerNorm`, `Softmax`, and `CrossEntropy`).

---

## Upcoming Book Roadmap

- [ ] **Part 0 · Foundations #3 — Backpropagation & Gradient Descent (Step-by-Step):** Visual, plain-English walkthrough of the chain rule and how `loss.backward()` computes gradients for every weight.
- [ ] **Part 0 · Foundations #4 — Self-Attention & Transformers in Plain English:** Demystify Queries, Keys, Values (`Q, K, V`), attention heatmaps, and how Mini-Llama predicts the next token.
- [ ] **Part I · End-to-End #5 & #6 — Imbalanced Classification (Telco Churn / Fraud):** Precision, Recall, F1, threshold tuning, Classical ML vs. `vktorch` MLP with class weighting.
- [ ] **Part II · Vulkan GPU Puzzles 03–06:** 2D Maps, Matrix Multiplication (tiling in shared memory), Parallel Prefix Sum, and Reductions via `@vnp.kernel`.
- [ ] **Publish Parts II–VI to the Online Book (`docs/`):** Export the GPU Puzzles, Feature Selection, UMAP, and Mini-Llama notebooks into the SynthWave '84 GitHub Pages site with full glossary popovers.

---

## Upcoming Vulkan Toolkit Roadmap

### Databases & Data Engines (`vkvec`, `vkgraph`, `vktime`, `vksql`)
- [x] **Full-Text Search (BM25) + Hybrid Search (`vkvec`):** Okapi `BM25Index` inverted index + `reciprocal_rank_fusion` (RRF) + `HybridIndex` combining exact keyword/code matching with Vulkan GPU dense vector search.
- [x] **Graph Database & GPU Graph Analytics (`vkgraph`):** In-VRAM `Graph` (edge lists + `vkpandas` DataFrames), GPU `pagerank`, `BFS` `shortest_path`, `GraphRAG` `k_hop_neighbors`, GPU `label_propagation`, `normalized_adjacency` for `vktorch` GCNs, `render_graph`, and Ch. 9–10!
- [x] **Time-Series Database & Signal Engine (`vktime`):** GPU `TimeSeries` (resampling, linear/rolling detrending, persistence), rolling windows, `make_lag_matrix`, `sliding_windows`, leak-free `temporal_split`, GPU `autocorrelation`, `periodogram`, `seasonal_decompose`, robust `detect_anomalies`, & `phase_fold` *(ready for Ch. 7 & 8!)*.
- [ ] **SQL Query Layer (`vksql` on `vkpandas`):** DuckDB-style `SELECT ... WHERE ... JOIN ... GROUP BY` SQL execution directly on GPU DataFrames.
- [ ] **Document / Key-Value (NoSQL) Guide:** Plain-English book sidebar comparing Relational, Columnar, Vector, Graph, Time-Series, and Document/KV stores so learners know which database fits each AI workload.

### `vktorch` Deep Learning
- [x] **Modern LLM Building Blocks:** Single-pass SPIR-V Rotary Positional Embeddings (**RoPE**), VRAM **KV-Cache** (`nn.KVCache` + `LlamaBlock(use_rope=True)`), and **LoRA** low-rank fine-tuning adapters (`nn.LoRALinear` + `.merge_weights()`).
- [x] **Fused SPIR-V `Conv2d` (`im2col` / `col2im`) & `MaxPool2d`:** 100% VRAM shaders (`12.8×` faster `im2col`/`col2im`, cutting 15-epoch Fashion-MNIST CNN training from `46.4s` → `24.5s`).
- [ ] **Mixed-Precision (`FP16`) Training & Gradient Accumulation** helpers for larger models on modest GPUs.

### `vknumpy` & Core Vulkan Backend
- [ ] **`vknumpy.fft` (`VkFFT`):** Header-only GPU 1D/2D FFTs for signal processing and physics demos.
- [ ] **Multi-GPU & Explicit Device Selection:** `vknumpy.list_devices()` and `vknumpy.set_device(idx)` for multi-GPU desktops/laptops (iGPU + dGPU).
- [ ] **Lazy Graph Fusion (`with vnp.fuse():`)** and zero-copy VRAM sharing between `ggml-vulkan` and `ncnn` on discrete GPUs.
- [ ] **Cross-Platform Wheels:** Pre-built manylinux, Windows, and macOS (MoltenVK) wheels via GitHub Actions.
