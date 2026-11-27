# `vkdata` Interactive Book Notebooks

Every folder below mirrors the **[Online Book](https://billyblackburn.github.io/vkdata/)** chapter-for-chapter — 100% pure Vulkan GPU, zero NumPy, zero CUDA/ROCm.

## Part 0 · AI Foundations (`00_foundations/`)
| Notebook | Topic |
|---|---|
| [`00_foundations/00a_the_perceptron.ipynb`](00_foundations/00a_the_perceptron.ipynb) | How a single artificial neuron works (`AND`/`OR`/`NOT`/`NAND`/`NOR`, learning a line, hitting `XOR`) |
| [`00_foundations/00b_multi_layer_perceptron.ipynb`](00_foundations/00b_multi_layer_perceptron.ipynb) | Bending space with `GELU`, solving `XOR` & concentric circles, and the 3 → 8B parameter ladder |

## AI by Data Type
| Folder | Data Type | Part 1 · Classical ML | Part 2 · Deep Learning (`vktorch`) |
|---|---|---|---|
| **`01_tabular/`** | **Tabular** *(California Housing)* | [`01_housing_classical_ml.ipynb`](01_tabular/01_housing_classical_ml.ipynb) | [`02_housing_neural_network.ipynb`](01_tabular/02_housing_neural_network.ipynb) |
| **`02_images/`** | **Images** *(Fashion-MNIST)* | [`03_fashion_classical_ml.ipynb`](02_images/03_fashion_classical_ml.ipynb) | [`04_fashion_cnn_deep_learning.ipynb`](02_images/04_fashion_cnn_deep_learning.ipynb) |
| **`03_vectors_and_text/`** | **Vectors & Text** *(BBC News)* | [`05_vector_search_classical_ml.ipynb`](03_vectors_and_text/05_vector_search_classical_ml.ipynb) | [`06_vector_embeddings_neural_net.ipynb`](03_vectors_and_text/06_vector_embeddings_neural_net.ipynb) |
| **`04_time_series/`** | **Time-Series** *(277-Yr Sunspots)* | [`07_solar_timeseries_classical_ml.ipynb`](04_time_series/07_solar_timeseries_classical_ml.ipynb) | [`08_solar_timeseries_neural_net.ipynb`](04_time_series/08_solar_timeseries_neural_net.ipynb) |
| **`05_graphs/`** | **Graphs** *(Karate Club & Cora)* | [`09_graph_classical_ml.ipynb`](05_graphs/09_graph_classical_ml.ipynb) | [`10_graph_neural_network_gcn.ipynb`](05_graphs/10_graph_neural_network_gcn.ipynb) |
| **`06_geospatial/`** | **Geospatial** *(USGS Earthquakes)* | [`01_earthquakes_classical_ml.ipynb`](06_geospatial/01_earthquakes_classical_ml.ipynb) | [`02_earthquakes_geo_neural_net.ipynb`](06_geospatial/02_earthquakes_geo_neural_net.ipynb) |

## Under the Hood · Vulkan GPU Puzzles (`00_vulkan_gpu_puzzles/`)
Write real GPU compute kernels in plain Python (`@vnp.kernel` → GLSL 450 → SPIR-V):
- [`puzzle_01_map.ipynb`](00_vulkan_gpu_puzzles/puzzle_01_map.ipynb)
- [`puzzle_01b_workgroup_sizes.ipynb`](00_vulkan_gpu_puzzles/puzzle_01b_workgroup_sizes.ipynb)
- [`puzzle_02_zip.ipynb`](00_vulkan_gpu_puzzles/puzzle_02_zip.ipynb)

## Bonus Labs (`bonus_labs/`)
- **`bonus_labs/package_demos/`** — Focused demos for `vknumpy`, `vkpandas`, `vkscipy`, `vkplot`, `vkpanel`, `vkcolorcet`, `vkshader`, `vkspatial`, `vkgeoviews`, and `vkvis`.
- **`bonus_labs/3d_and_deep_dives/`** — 3D GPU showcases (galaxies, ocean FFT, proteins, earthquakes, OpenSky flights) and deep-learning labs (backprop, Mini-GPT, Mini-Llama).
