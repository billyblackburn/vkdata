# `vkdata` Book & Vulkan GPU Toolkit Roadmap

This roadmap tracks what is complete and what's next for both the **interactive follow-along book** ([billyblackburn.github.io/vkdata](https://billyblackburn.github.io/vkdata/)) and the bundled **pure-Vulkan GPU toolkit** (`vkdata`).

---

## Completed Milestones

### Interactive Book & Learning Platform
- **Online SynthWave '84 Book (`docs/`):** Mobile-friendly HTML chapters with Python syntax highlighting, sticky Prev/Next navigation, **52-term searchable Glossary** ([`docs/glossary.html`](docs/glossary.html)), and **in-chapter tap-to-explain acronym popovers** (`200+` links across prose).
- **Part 0 · AI Foundations (`examples/00_ai_foundations/`):**
  - *The Perceptron:* All 5 basic single-neuron logic gates (`AND`, `OR`, `NOT`, `NAND`, `NOR`), `vklearn.Perceptron`, Vulkan GPU smooth neuron, & the `XOR` wall.
  - *Multi-Layer Perceptrons (MLPs):* Solving `XOR`, why `GELU`/`ReLU` activations prevent linear collapse, 1 Neuron vs. Tiny MLP vs. Deep MLP on concentric circles, & the 3-to-8B LLM parameter ladder.
- **Part I · End-to-End Projects (`examples/tldr_end2end/`):**
  - *California Housing (Regression):* Classical ML (`$69.6k` → `$52.7k` GPU KNN) vs. `vktorch` Residual MLP (**`$48.2k` RMSE**, `R² = 0.828`).
  - *Fashion-MNIST (Images):* Classical CV (`81.50%` → `86.45%` PCA + Edge/HOG) vs. `vktorch` 2D CNN (**`87.85%`**, `Shirt` `+11.5 pts`, 46s on Steam Deck APU).
- **Parts II–VI Notebooks (`examples/`):** Vulkan GPU Puzzles (`@vnp.kernel`), GPU Foundations, Visual EDA on 11.84M rows, Feature Selection & 7-algorithm UMAP/Manifold Learning, Classical ML at scale, Backprop Lab, Autoencoders, & Mini-Llama (`RMSNorm` + `SwiGLU`).
- **Hardware Inspector:** [`scripts/gpu_specs.py`](scripts/gpu_specs.py) (`pixi run gpu-specs`) for instant Vulkan device limits + 1-second FP32 GEMM & SPIR-V bandwidth check.

### Bundled Pure-Vulkan Toolkit (14 Packages · 72/72 Tests Passing)
- **Core Compute & Data:** `vknumpy` (`ggml-vulkan` + `ncnn` + `@vnp.kernel` Python-to-SPIR-V compiler), `vkpandas` (full GPU DataFrame, filtering, `groupby`, `merge`, Parquet/CSV), `vkscipy`, `vkspatial` (Morton 2D `GeoDataFrame`, `sjoin`, 6-level 3D Barnes-Hut Octree), `vkastro`.
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

### `vktorch` Deep Learning
- [ ] **Modern LLM Building Blocks:** Rotary Positional Embeddings (**RoPE**), **KV-Cache** for fast autoregressive decoding, and **LoRA** low-rank fine-tuning adapters.
- [x] **Fused SPIR-V `Conv2d` (`im2col` / `col2im`) & `MaxPool2d`:** 100% VRAM shaders (`12.8×` faster `im2col`/`col2im`, cutting 15-epoch Fashion-MNIST CNN training from `46.4s` → `24.5s`).
- [ ] **Mixed-Precision (`FP16`) Training & Gradient Accumulation** helpers for larger models on modest GPUs.

### `vknumpy` & Core Vulkan Backend
- [ ] **`vknumpy.fft` (`VkFFT`):** Header-only GPU 1D/2D FFTs for signal processing and physics demos.
- [ ] **Multi-GPU & Explicit Device Selection:** `vknumpy.list_devices()` and `vknumpy.set_device(idx)` for multi-GPU desktops/laptops (iGPU + dGPU).
- [ ] **Lazy Graph Fusion (`with vnp.fuse():`)** and zero-copy VRAM sharing between `ggml-vulkan` and `ncnn` on discrete GPUs.
- [ ] **Cross-Platform Wheels:** Pre-built manylinux, Windows, and macOS (MoltenVK) wheels via GitHub Actions.
