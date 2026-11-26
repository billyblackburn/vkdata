"""Generate docs/reference.html: Live, searchable SynthWave '84 API Reference for all 16 vkdata packages."""
from __future__ import annotations

import html as html_lib
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Curated metadata for all 16 packages:
# (pkg_name, category, accent_hex, equivalent, summary, example_code, extra_submodules)
PACKAGES: List[Tuple[str, str, str, str, str, str, List[str]]] = [
    (
        "vknumpy",
        "Core Compute & Data",
        "#36f9f6",
        "NumPy / CuPy",
        "Core Vulkan GPU ndarray engine backed by ggml-vulkan, ncnn, and the @vnp.kernel Python-to-SPIR-V compiler. All 15 other packages share its zero-copy VkBuffers.",
        "import vknumpy as vnp\na = vnp.array([[1.0, 2.0], [3.0, 4.0]])\nprint((a @ a.T).to_numpy())",
        ["vknumpy.linalg"],
    ),
    (
        "vkpandas",
        "Core Compute & Data",
        "#36f9f6",
        "pandas / cuDF",
        "GPU-resident DataFrame and Series for tabular data: filtering, groupby aggregations, inner joins, sorting, rolling windows, and Parquet/CSV I/O directly in VRAM.",
        "import vkpandas as vpd\ndf = vpd.DataFrame({'city': ['LA', 'SF', 'LA'], 'price': [500, 900, 650]})\nprint(df.groupby('city')['price'].mean())",
        [],
    ),
    (
        "vkscipy",
        "Core Compute & Data",
        "#36f9f6",
        "SciPy",
        "GPU scientific computing submodules: pairwise distances (spatial), correlation & z-scores (stats), special math functions, 2D image filters (ndimage), signals, and curve fitting (optimize).",
        "from vkscipy import stats, ndimage\ncorr = stats.corrcoef(X_gpu)\nedges = ndimage.sobel(img_gpu)",
        [
            "vkscipy.stats",
            "vkscipy.spatial",
            "vkscipy.special",
            "vkscipy.ndimage",
            "vkscipy.signal",
            "vkscipy.optimize",
        ],
    ),
    (
        "vklearn",
        "AI, ML, Search & Time-Series",
        "#72f1b8",
        "scikit-learn / cuML",
        "Classical Machine Learning on Vulkan GPU: scalers, LinearRegression, Ridge, LogisticRegression, Perceptron, PCA, KMeans, KNN, Pipelines, and evaluation metrics.",
        "import vklearn as vkl\nmodel = vkl.Ridge(alpha=1.0).fit(X_tr, y_tr)\npreds = model.predict(X_te)",
        [],
    ),
    (
        "vktorch",
        "AI, ML, Search & Time-Series",
        "#72f1b8",
        "PyTorch",
        "Dynamic reverse-mode autograd and deep learning framework with fused single-pass SPIR-V kernels (LayerNorm, RMSNorm, GELU, SiLU, SwiGLU, Softmax, CrossEntropy, AdamW, Conv2d im2col, MaxPool2d, RoPE, KV-Cache, LoRA).",
        "import vktorch as vt, vktorch.nn as nn, vktorch.functional as F\nlayer = nn.Linear(16, 4)\nloss = F.cross_entropy(layer(vt.Tensor(X)), y)\nloss.backward()",
        ["vktorch.nn", "vktorch.functional", "vktorch.optim"],
    ),
    (
        "vkvec",
        "AI, ML, Search & Time-Series",
        "#72f1b8",
        "FAISS / Qdrant / ChromaDB",
        "In-VRAM Vulkan GPU Vector Database & Hybrid Search engine: Cosine/L2/Dot VectorIndex with metadata filters, Okapi BM25Index full-text inverted index, Reciprocal Rank Fusion (HybridIndex), GPU TfidfVectorizer, and RAG chunk_text.",
        "import vkvec\nhyb = vkvec.HybridIndex(metric='cosine').add(embs, docs)\nhits = hyb.search(q_emb, 'Sony PlayStation', k=5, mode='rrf')",
        [],
    ),
    (
        "vktime",
        "AI, ML, Search & Time-Series",
        "#ff9e64",
        "Timescale / statsmodels",
        "GPU Time-Series database & signal toolkit: TimeSeries container (resampling, detrending, persistence), rolling stats & EWMA, supervised lag matrices, 1D-CNN sliding windows, leak-free temporal_split, GPU autocorrelation, periodogram, anomaly detection, and astronomy phase_fold.",
        "import vktime\nX, y = vktime.make_lag_matrix(series, lags=12, include_rolling=[6, 12])\nX_tr, X_te, y_tr, y_te = vktime.temporal_split(X, y, train_ratio=0.8)",
        [],
    ),
    (
        "vkvis",
        "AI, ML, Search & Time-Series",
        "#72f1b8",
        "UMAP / mlx-vis",
        "Pure-Vulkan GPU manifold learning and dimensionality reduction (UMAP, TSNE, PaCMAP, LocalMAP, TriMap, DREAMS, CNE, NNDescent) plus crisp 2D/3D embedding renderers.",
        "import vkvis\nY = vkvis.UMAP(n_neighbors=15, metric='cosine', seed=42).fit_transform(X)\nimg = vkvis.render_embedding(Y, labels=y)",
        [],
    ),
    (
        "vkshader",
        "Visualization & Dashboards",
        "#ff7edb",
        "Datashader (2D + 3D)",
        "Zero-copy SPIR-V 2D/3D rasterizer for millions of points, lines, and spheres — with 0.91 ms 3-mip HDR Bloom + ACES tonemapping, Esri map tiles, and interactive anywidget viewers.",
        "import vkshader as vks\ncvs = vks.Canvas(plot_width=800, plot_height=500)\nimg = vks.shade(cvs.points(df, 'x', 'y'), cmap='fire')",
        [],
    ),
    (
        "vkcolorcet",
        "Visualization & Dashboards",
        "#ff7edb",
        "Colorcet",
        "26+ perceptually uniform scientific colormaps cached as 256-entry GPU lookup tables (LUTs) for accurate, artifact-free visualizations.",
        "import vkcolorcet as vcc\nlut = vcc.get_rgb_lut('fire')",
        [],
    ),
    (
        "vkholoviews",
        "Visualization & Dashboards",
        "#ff7edb",
        "HoloViews",
        "Declarative high-level plotting with Points, Path, Curve, Histogram, Bars, and Tiles — compose overlays with * and side-by-side layouts with +.",
        "import vkholoviews as vhv\nplot = vhv.Points(df, kdims=['x', 'y']) * vhv.Curve(line_df)",
        [],
    ),
    (
        "vkgeoviews",
        "Visualization & Dashboards",
        "#ff7edb",
        "GeoViews",
        "Coordinate-system-aware geospatial plotting (Points, Path, Polygons choropleths) layered over live map tile_sources.",
        "import vkgeoviews as vgv\nmap_view = vgv.tile_sources.EsriDark * vgv.Points(gdf)",
        [],
    ),
    (
        "vkplot",
        "Visualization & Dashboards",
        "#ff7edb",
        "hvPlot",
        "One-line .vkplot EDA plotting accessor registered on vkpandas DataFrames, Series, and vkspatial GeoDataFrames.",
        "import vkplot\ndf.vkplot.scatter(x='longitude', y='latitude', c='price')",
        [],
    ),
    (
        "vkpanel",
        "Visualization & Dashboards",
        "#ff7edb",
        "Panel",
        "Reactive notebook dashboards with Row, Column, Card, Tabs, DataTable, sliders/select widgets, and pn.bind(...) live cross-filtering.",
        "import vkpanel as vpn\nslider = vpn.IntSlider(name='K', start=1, end=25, value=5)\nvpn.Column(slider, vpn.bind(update_plot, slider))",
        [],
    ),
    (
        "vkspatial",
        "Geospatial & Astronomy",
        "#fede5d",
        "GeoPandas / cuSpatial",
        "Zero-copy SPIR-V 2D Morton-curve GeoDataFrame, single-pass GPU spatial joins (sjoin), point_in_polygon, Haversine distances, GeoJSON I/O, and 6-level 3D Barnes-Hut Octree.",
        "import vkspatial as vsp\ndist_km = vsp.haversine(lat1, lon1, lat2, lon2)",
        [],
    ),
    (
        "vkastro",
        "Geospatial & Astronomy",
        "#fede5d",
        "Astropy",
        "GPU astronomy toolkit: SkyCoord equatorial/galactic transforms, angular separations, Aitoff sky projections, photometry magnitudes, FlatLambdaCDM cosmology, and FITS reader.",
        "import vkastro as vas\ncoords = vas.SkyCoord(ra=ra_deg, dec=dec_deg)\nx, y = vas.aitoff_projection(ra_deg, dec_deg)",
        [],
    ),
]

# Plain-English fallback summaries for standard NumPy-style math & array helpers without docstrings
FALLBACK_DOCS: Dict[str, str] = {
    "abs": "Element-wise absolute value |x| on Vulkan GPU.",
    "absolute": "Element-wise absolute value |x| on Vulkan GPU.",
    "add": "Element-wise GPU addition (a + b) with broadcasting.",
    "all": "Test whether all array elements evaluate to True along an axis.",
    "any": "Test whether any array element evaluates to True along an axis.",
    "arange": "Create evenly spaced values within a given interval on GPU.",
    "arccos": "Element-wise inverse cosine (arccos) on GPU.",
    "arccosh": "Element-wise inverse hyperbolic cosine on GPU.",
    "arcsin": "Element-wise inverse sine (arcsin) on GPU.",
    "arcsinh": "Element-wise inverse hyperbolic sine on GPU.",
    "arctan": "Element-wise inverse tangent (arctan) on GPU.",
    "arctan2": "Element-wise two-argument arctangent atan2(y, x) on GPU.",
    "arctanh": "Element-wise inverse hyperbolic tangent on GPU.",
    "argmax": "Return indices of maximum values along an axis on GPU.",
    "argmin": "Return indices of minimum values along an axis on GPU.",
    "argsort": "Return indices that sort an array along an axis on GPU.",
    "array": "Create a Vulkan GPU ndarray from a Python list or NumPy array.",
    "asarray": "Convert input to a Vulkan GPU ndarray without copying if already on GPU.",
    "atleast_1d": "View input as an array with at least one dimension.",
    "atleast_2d": "View input as an array with at least two dimensions.",
    "bincount": "Count occurrences of each non-negative integer value via atomic SPIR-V shader.",
    "broadcast_to": "Broadcast an array to a new compatible shape.",
    "cbrt": "Element-wise cube root on GPU.",
    "ceil": "Element-wise ceiling (round up to integer) on GPU.",
    "clip": "Limit (clamp) array values between a_min and a_max on GPU.",
    "column_stack": "Stack 1D arrays as columns into a 2D GPU array.",
    "concatenate": "Join a sequence of GPU arrays along an existing axis.",
    "copy": "Return an independent VRAM copy of a GPU array.",
    "cos": "Element-wise cosine on GPU.",
    "cosh": "Element-wise hyperbolic cosine on GPU.",
    "count_nonzero": "Count the number of non-zero elements in an array.",
    "cross": "Compute the 3D vector cross product on GPU.",
    "cumprod": "Return the cumulative product of elements along an axis.",
    "cumsum": "Return the cumulative sum of elements along an axis on GPU.",
    "deg2rad": "Convert angles from degrees to radians on GPU.",
    "degrees": "Convert angles from radians to degrees on GPU.",
    "diag": "Extract a diagonal or construct a diagonal 2D matrix.",
    "diagonal": "Return specified diagonals of an array.",
    "diff": "Calculate the n-th discrete difference along an axis.",
    "divide": "Element-wise GPU division (a / b) with broadcasting.",
    "dot": "Dot product or matrix multiplication of two GPU arrays.",
    "einsum": "Einstein summation convention for tensor contractions on GPU.",
    "empty": "Allocate an uninitialized GPU array of the given shape.",
    "empty_like": "Allocate an uninitialized GPU array matching another array's shape.",
    "equal": "Element-wise equality comparison (a == b) on GPU.",
    "exp": "Element-wise exponential e^x on GPU.",
    "exp2": "Element-wise base-2 exponential 2^x on GPU.",
    "expand_dims": "Insert a new length-1 axis into an array's shape.",
    "expm1": "Element-wise exp(x) - 1 with high accuracy near zero.",
    "eye": "Create a 2D identity matrix with ones on the diagonal.",
    "fabs": "Element-wise floating-point absolute value on GPU.",
    "flip": "Reverse the order of elements along an axis.",
    "floor": "Element-wise floor (round down to integer) on GPU.",
    "floor_divide": "Element-wise integer floor division (a // b).",
    "fmax": "Element-wise maximum ignoring NaNs.",
    "fmin": "Element-wise minimum ignoring NaNs.",
    "fmod": "Element-wise remainder of division.",
    "full": "Create a GPU array filled with a constant scalar value.",
    "full_like": "Create a constant-filled GPU array matching another array's shape.",
    "gelu": "Gaussian Error Linear Unit activation function on GPU.",
    "greater": "Element-wise greater-than comparison (a > b).",
    "greater_equal": "Element-wise greater-than-or-equal comparison (a >= b).",
    "histogram": "Compute 1D histogram bin counts on GPU.",
    "histogram2d": "Fast atomic SPIR-V 2D histogram on Vulkan GPU.",
    "hstack": "Stack arrays horizontally (column-wise).",
    "hypot": "Element-wise Euclidean hypotenuse sqrt(x1^2 + x2^2).",
    "identity": "Create a square identity matrix on GPU.",
    "inner": "Inner product of two arrays.",
    "isfinite": "Test element-wise for finiteness (not NaN or infinity).",
    "isinf": "Test element-wise for positive or negative infinity.",
    "isnan": "Test element-wise for NaN (Not a Number).",
    "isclose": "Return boolean array where two arrays are element-wise equal within tolerance.",
    "kron": "Kronecker product of two arrays.",
    "less": "Element-wise less-than comparison (a < b).",
    "less_equal": "Element-wise less-than-or-equal comparison (a <= b).",
    "linspace": "Create evenly spaced numbers over a specified interval on GPU.",
    "log": "Element-wise natural logarithm ln(x) on GPU.",
    "log10": "Element-wise base-10 logarithm on GPU.",
    "log1p": "Element-wise log(1 + x) with high accuracy near zero.",
    "log2": "Element-wise base-2 logarithm on GPU.",
    "logaddexp": "Logarithm of the sum of exponentiations log(exp(x1) + exp(x2)).",
    "logical_and": "Element-wise logical AND.",
    "logical_not": "Element-wise logical NOT.",
    "logical_or": "Element-wise logical OR.",
    "logical_xor": "Element-wise logical XOR.",
    "logspace": "Create numbers spaced evenly on a log scale.",
    "matmul": "Vulkan GPU matrix multiplication (a @ b) via ggml-vulkan.",
    "max": "Return maximum value of an array or along an axis on GPU.",
    "maximum": "Element-wise maximum of two GPU arrays.",
    "mean": "Compute arithmetic mean along an axis on GPU.",
    "median": "Compute median along an axis.",
    "meshgrid": "Create 2D/ND coordinate grids from 1D coordinate vectors.",
    "min": "Return minimum value of an array or along an axis on GPU.",
    "minimum": "Element-wise minimum of two GPU arrays.",
    "mod": "Element-wise remainder of division (a % b).",
    "moveaxis": "Move axes of an array to new positions.",
    "multiply": "Element-wise GPU multiplication (a * b) with broadcasting.",
    "nan_to_num": "Replace NaN with zero and infinity with large finite numbers.",
    "nanmax": "Maximum of an array ignoring NaNs.",
    "nanmean": "Arithmetic mean along an axis ignoring NaNs.",
    "nanmin": "Minimum of an array ignoring NaNs.",
    "nanstd": "Standard deviation along an axis ignoring NaNs.",
    "nansum": "Sum of elements along an axis treating NaNs as zero.",
    "nanvar": "Variance along an axis ignoring NaNs.",
    "negative": "Element-wise numerical negation (-x) on GPU.",
    "nonzero": "Return indices of non-zero elements via GPU stream compaction.",
    "not_equal": "Element-wise inequality comparison (a != b).",
    "ones": "Create a GPU array filled with ones.",
    "ones_like": "Create a ones-filled GPU array matching another array's shape.",
    "outer": "Compute the outer product of two vectors on GPU.",
    "pad": "Pad an array with constant or edge values.",
    "percentile": "Compute the q-th percentile of the data along an axis.",
    "power": "Element-wise exponentiation (a ** b) on GPU.",
    "prod": "Product of array elements over a given axis.",
    "ptp": "Peak-to-peak range (max - min) along an axis.",
    "quantile": "Compute the q-th quantile of the data along an axis.",
    "rad2deg": "Convert angles from radians to degrees on GPU.",
    "radians": "Convert angles from degrees to radians on GPU.",
    "ravel": "Return a contiguous flattened 1D array.",
    "reciprocal": "Element-wise reciprocal (1 / x) on GPU.",
    "relu": "Rectified Linear Unit activation max(0, x) on GPU.",
    "remainder": "Element-wise remainder of division.",
    "repeat": "Repeat elements of an array.",
    "reshape": "Give a new shape to a GPU array without changing its data.",
    "rint": "Round elements to the nearest integer.",
    "roll": "Roll array elements along a given axis.",
    "round": "Evenly round array elements to a given number of decimals.",
    "searchsorted": "Find indices where elements should be inserted to maintain sorted order.",
    "sigmoid": "Element-wise logistic sigmoid 1 / (1 + exp(-x)) on GPU.",
    "sign": "Element-wise sign indicator (-1, 0, or +1).",
    "silu": "Sigmoid Linear Unit (Swish) activation x * sigmoid(x) on GPU.",
    "sin": "Element-wise sine on GPU.",
    "sinh": "Element-wise hyperbolic sine on GPU.",
    "softmax": "Numerically stable Softmax probabilities along an axis on GPU.",
    "sort": "Return a sorted copy of a GPU array along an axis.",
    "split": "Split an array into multiple sub-arrays.",
    "sqrt": "Element-wise square root on GPU.",
    "square": "Element-wise square (x^2) on GPU.",
    "squeeze": "Remove length-1 axes from an array's shape.",
    "stack": "Join a sequence of arrays along a new axis.",
    "std": "Compute standard deviation along an axis on GPU.",
    "subtract": "Element-wise GPU subtraction (a - b) with broadcasting.",
    "sum": "Sum of array elements along an axis on GPU.",
    "swapaxes": "Interchange two axes of an array.",
    "take": "Gather elements from an array along an axis by integer indices.",
    "tan": "Element-wise tangent on GPU.",
    "tanh": "Element-wise hyperbolic tangent on GPU.",
    "tensordot": "Compute tensor dot product along specified axes on GPU.",
    "tile": "Construct an array by repeating A along each axis.",
    "top_k": "Return indices of the top-k largest elements along an axis on GPU.",
    "trace": "Return the sum along diagonals of a 2D array.",
    "transpose": "Permute the dimensions of a GPU array.",
    "tri": "Lower-triangular matrix of ones.",
    "tril": "Lower triangle of a 2D matrix.",
    "triu": "Upper triangle of a 2D matrix.",
    "trunc": "Element-wise truncation toward zero.",
    "unique": "Find the sorted unique elements of an array.",
    "var": "Compute variance along an axis on GPU.",
    "vdot": "Dot product of two flattened vectors.",
    "vstack": "Stack arrays vertically (row-wise).",
    "where": "Return elements chosen from x or y depending on condition.",
    "zeros": "Create a GPU array filled with zeros.",
    "zeros_like": "Create a zeros-filled GPU array matching another array's shape.",
    # linalg
    "cholesky": "Cholesky decomposition of a symmetric positive-definite matrix.",
    "det": "Compute the determinant of a square matrix.",
    "eig": "Compute eigenvalues and right eigenvectors of a square matrix.",
    "eigh": "Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix.",
    "inv": "Compute the multiplicative inverse of a square matrix.",
    "lstsq": "Return the least-squares solution to a linear matrix equation.",
    "matrix_power": "Raise a square matrix to an integer power.",
    "matrix_rank": "Return matrix rank using SVD.",
    "norm": "Compute vector or matrix norm (L2, L1, Frobenius, etc.) on GPU.",
    "pinv": "Compute the Moore-Penrose pseudo-inverse of a matrix.",
    "qr": "Compute the QR factorization of a matrix.",
    "slogdet": "Compute the sign and natural log of the determinant.",
    "solve": "Solve a linear matrix equation Ax = b.",
    "solve_triangular": "Solve a triangular linear system Tx = b.",
    "svd": "Singular Value Decomposition (U, S, Vt) of a matrix.",
}


def _short_doc(obj: Any, name: str) -> str:
    doc = (inspect.getdoc(obj) or "").strip()
    if doc:
        # Take first non-empty paragraph, collapse whitespace
        para_lines = []
        for line in doc.splitlines():
            s = line.strip()
            if not s:
                if para_lines:
                    break
                continue
            if s.startswith(("Parameters", "Returns", "Examples", "---", "===")):
                break
            para_lines.append(s)
        text = " ".join(para_lines)
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        return text
    if name in FALLBACK_DOCS:
        return FALLBACK_DOCS[name]
    if inspect.isclass(obj):
        return f"{name} class."
    return f"GPU-accelerated {name} operation."


def _safe_sig(obj: Any) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "(...)"
    # Clean overly verbose typing annotations for readability
    sig = sig.replace("typing.", "").replace("numpy.", "np.").replace("vknumpy.ndarray.ndarray", "vnp.ndarray")
    if len(sig) > 110:
        sig = sig[:107] + "...)"
    return sig


def _collect_module_entries(mod_name: str, pkg_root: str) -> List[Dict[str, Any]]:
    mod = importlib.import_module(mod_name)
    has_all = hasattr(mod, "__all__")
    names = list(getattr(mod, "__all__", sorted(k for k in dir(mod) if not k.startswith("_"))))
    prefix = "" if mod_name == pkg_root else mod_name[len(pkg_root) + 1 :] + "."

    entries: List[Dict[str, Any]] = []
    seen = set()
    for name in names:
        if name.startswith("_") or name in seen:
            continue
        obj = getattr(mod, name, None)
        if obj is None or inspect.ismodule(obj):
            continue
        if not (inspect.isclass(obj) or callable(obj)):
            continue
        # If no __all__, filter out imported typing/stdlib helpers
        if not has_all:
            obj_mod = getattr(obj, "__module__", "") or ""
            if not obj_mod.startswith(pkg_root):
                continue
        seen.add(name)

        kind = "class" if inspect.isclass(obj) else "func"
        methods: List[Dict[str, str]] = []
        if kind == "class":
            for mname, mobj in inspect.getmembers(obj):
                if mname.startswith("_"):
                    continue
                if inspect.isfunction(mobj) or inspect.ismethod(mobj) or isinstance(
                    inspect.getattr_static(obj, mname, None), (classmethod, staticmethod, property)
                ):
                    raw = inspect.getattr_static(obj, mname, mobj)
                    if isinstance(raw, property):
                        msig = " (property)"
                        mdoc = _short_doc(raw.fget, mname) if raw.fget else f"{mname} property."
                    else:
                        msig = _safe_sig(mobj)
                        mdoc = _short_doc(mobj, mname)
                    methods.append({"name": mname, "sig": msig, "doc": mdoc})

        entries.append(
            {
                "display": f"{prefix}{name}",
                "name": name,
                "kind": kind,
                "sig": _safe_sig(obj),
                "doc": _short_doc(obj, name),
                "methods": methods,
            }
        )
    # Sort classes first, then functions alphabetically
    entries.sort(key=lambda e: (0 if e["kind"] == "class" else 1, e["display"].lower()))
    return entries


def build_reference() -> Tuple[int, int]:
    pkg_blocks: List[str] = []
    jump_pills: List[str] = []
    total_entries = 0
    total_methods = 0

    for pkg_name, category, accent, equiv, summary, example, extras in PACKAGES:
        entries = _collect_module_entries(pkg_name, pkg_name)
        for sub in extras:
            entries.extend(_collect_module_entries(sub, pkg_name))

        total_entries += len(entries)
        jump_pills.append(
            f'<a class="pkg-pill" href="#pkg-{pkg_name}" style="--accent:{accent};">'
            f'<code>{pkg_name}</code> <span class="cnt">{len(entries)}</span></a>'
        )

        # Compact vs detailed rendering: for huge math modules (>40 funcs), group functions cleanly
        cards_html: List[str] = []
        for e in entries:
            kind_badge = (
                '<span class="kind-badge kind-cls">CLASS</span>'
                if e["kind"] == "class"
                else '<span class="kind-badge kind-fn">FUNC</span>'
            )
            methods_html = ""
            if e["methods"]:
                total_methods += len(e["methods"])
                m_rows = []
                for m in e["methods"]:
                    m_rows.append(
                        f'<div class="m-row" data-search="{html_lib.escape((pkg_name + " " + e["display"] + "." + m["name"] + " " + m["doc"]).lower())}">'
                        f'<code class="m-name">.{html_lib.escape(m["name"])}</code>'
                        f'<span class="m-sig">{html_lib.escape(m["sig"])}</span>'
                        f'<div class="m-doc">{html_lib.escape(m["doc"])}</div>'
                        f"</div>"
                    )
                methods_html = (
                    f'<details class="m-box"><summary>Show {len(e["methods"])} methods &amp; properties</summary>'
                    f'<div class="m-list">{"".join(m_rows)}</div></details>'
                )

            search_blob = f"{pkg_name} {category} {equiv} {e['display']} {e['sig']} {e['doc']} " + " ".join(
                m["name"] for m in e["methods"]
            )
            cards_html.append(
                f'<div class="api-item" data-search="{html_lib.escape(search_blob.lower())}">'
                f'<div class="api-head">{kind_badge}'
                f'<code class="api-name">{html_lib.escape(pkg_name)}.{html_lib.escape(e["display"])}</code>'
                f'<span class="api-sig">{html_lib.escape(e["sig"])}</span></div>'
                f'<p class="api-doc">{html_lib.escape(e["doc"])}</p>'
                f"{methods_html}</div>"
            )

        pkg_blocks.append(
            f'<section class="pkg-sec" id="pkg-{pkg_name}" style="--accent:{accent};">'
            f'<div class="pkg-head">'
            f'<div><span class="pkg-cat">{html_lib.escape(category)}</span>'
            f'<h2><code>{pkg_name}</code> <span class="pkg-eq">({html_lib.escape(equiv)})</span></h2></div>'
            f'<span class="pkg-count">{len(entries)} exported APIs</span>'
            f"</div>"
            f'<p class="pkg-sum">{html_lib.escape(summary)}</p>'
            f'<pre class="pkg-ex"><code>{html_lib.escape(example)}</code></pre>'
            f'<div class="api-grid">{"".join(cards_html)}</div>'
            f"</section>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>vkdata API Reference (16 Vulkan GPU Packages)</title>
  <style>
    :root {{
      --bg: #262335; --card: #1e1a2b; --border: #495495;
      --text: #f4eee4; --muted: #b6b1d6;
      --pink: #ff7edb; --cyan: #36f9f6; --yellow: #fede5d; --mint: #72f1b8; --peach: #ff9e64;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background-color: var(--bg); color: var(--text);
      background-image:
        radial-gradient(circle at 12% 0%, rgba(54, 249, 246, 0.14), transparent 38%),
        radial-gradient(circle at 88% 5%, rgba(255, 126, 219, 0.14), transparent 40%),
        linear-gradient(180deg, #241b2f 0%, #262335 45%, #1c1828 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.55; min-height: 100vh;
    }}
    .top-nav {{
      position: sticky; top: 0; z-index: 100;
      background: linear-gradient(90deg, #1b1528, #241b35, #1b1528);
      border-bottom: 1px solid #ff7edb;
      box-shadow: 0 2px 16px rgba(255, 126, 219, 0.22);
      padding: 10px 16px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px;
    }}
    .top-nav a {{
      color: var(--cyan); text-decoration: none; font-weight: 700; font-size: 13.5px;
      padding: 5px 11px; border-radius: 7px; background: #2a2139;
      border: 1px solid rgba(54, 249, 246, 0.45);
    }}
    .top-nav a.gloss {{ color: var(--yellow); border-color: rgba(254, 222, 93, 0.45); }}
    .top-nav a:hover {{ border-color: var(--pink); color: var(--pink); }}
    .wrap {{ max-width: 1040px; margin: 0 auto; padding: 24px 16px 64px; }}
    h1 {{
      margin: 0 0 8px; font-size: clamp(24px, 4.5vw, 36px);
      color: var(--cyan); text-shadow: 0 0 16px rgba(54, 249, 246, 0.45);
    }}
    .sub {{ color: var(--muted); margin: 0 0 18px; font-size: 15px; }}
    #search {{
      width: 100%; padding: 12px 15px; font-size: 15px; border-radius: 10px;
      background: #171422; color: var(--text); border: 1px solid var(--cyan);
      box-shadow: 0 0 14px rgba(54, 249, 246, 0.16); outline: none; margin-bottom: 14px;
    }}
    #search:focus {{ border-color: var(--pink); box-shadow: 0 0 18px rgba(255, 126, 219, 0.28); }}
    .pill-bar {{ display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 24px; }}
    .pkg-pill {{
      text-decoration: none; display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 10px; border-radius: 999px; font-size: 12.5px; font-weight: 700;
      background: #1b1728; border: 1px solid var(--accent); color: var(--accent);
      transition: transform 0.12s, background 0.12s;
    }}
    .pkg-pill:hover {{ transform: translateY(-1px); background: rgba(255, 255, 255, 0.08); }}
    .pkg-pill code {{ background: none; padding: 0; color: inherit; font-size: 12.5px; }}
    .pkg-pill .cnt {{
      font-size: 11px; color: #1b1728; background: var(--accent);
      padding: 0 6px; border-radius: 999px; font-weight: 800;
    }}
    .pkg-sec {{
      margin-bottom: 34px; background: rgba(27, 23, 40, 0.72);
      border: 1px solid var(--border); border-top: 3px solid var(--accent);
      border-radius: 14px; padding: 18px 18px 20px;
    }}
    .pkg-head {{
      display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap;
    }}
    .pkg-cat {{
      font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--accent);
    }}
    .pkg-sec h2 {{ margin: 2px 0 6px; font-size: 22px; color: #fff; }}
    .pkg-sec h2 code {{ color: var(--accent); background: none; padding: 0; font-size: 22px; }}
    .pkg-eq {{ font-size: 14px; font-weight: 500; color: var(--muted); }}
    .pkg-count {{
      font-size: 12px; font-weight: 700; color: var(--accent);
      border: 1px solid var(--accent); background: rgba(255,255,255,0.04);
      padding: 3px 10px; border-radius: 999px;
    }}
    .pkg-sum {{ margin: 0 0 12px; color: #ded7ee; font-size: 14.5px; }}
    .pkg-ex {{
      margin: 0 0 16px; padding: 11px 14px; border-radius: 9px; overflow-x: auto;
      background: #14111f; border: 1px solid rgba(73, 84, 149, 0.55);
      color: #72f1b8; font-size: 13px; line-height: 1.45;
    }}
    .api-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }}
    .api-item {{
      background: var(--card); border: 1px solid rgba(73, 84, 149, 0.55);
      border-left: 3px solid var(--accent); border-radius: 9px; padding: 10px 12px;
    }}
    .api-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px; margin-bottom: 4px; }}
    .kind-badge {{
      font-size: 10px; font-weight: 800; padding: 1px 6px; border-radius: 4px; letter-spacing: 0.04em;
    }}
    .kind-cls {{ background: rgba(254, 222, 93, 0.16); color: var(--yellow); border: 1px solid rgba(254, 222, 93, 0.5); }}
    .kind-fn  {{ background: rgba(54, 249, 246, 0.13); color: var(--cyan); border: 1px solid rgba(54, 249, 246, 0.45); }}
    .api-name {{ color: #ffffff; font-weight: 700; font-size: 13.5px; word-break: break-word; }}
    .api-sig {{ color: #ff9e64; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; word-break: break-word; }}
    .api-doc {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .m-box {{ margin-top: 8px; border-top: 1px dashed rgba(73, 84, 149, 0.5); padding-top: 6px; }}
    .m-box summary {{ cursor: pointer; color: var(--pink); font-size: 12px; font-weight: 700; }}
    .m-list {{ margin-top: 6px; display: flex; flex-direction: column; gap: 5px; }}
    .m-row {{ background: #171422; border-radius: 6px; padding: 5px 8px; }}
    .m-name {{ color: var(--mint); font-weight: 700; font-size: 12px; }}
    .m-sig {{ color: #ff9e64; font-family: ui-monospace, monospace; font-size: 11px; margin-left: 4px; }}
    .m-doc {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
    .match-status {{ color: var(--yellow); font-size: 13px; margin: -6px 0 14px; font-weight: 600; }}
  </style>
</head>
<body>
  <nav class="top-nav">
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <a href="index.html">&larr; Book Home</a>
      <a class="gloss" href="glossary.html">&#128214; Acronym Glossary</a>
    </div>
    <span style="color:#ff7edb;font-weight:700;font-size:13.5px;">vkdata API Reference &middot; 16 Packages</span>
  </nav>

  <div class="wrap">
    <h1><code>vkdata</code> Toolset API Reference</h1>
    <p class="sub">
      Auto-generated directly from the live code across all <strong>16 pure-Vulkan GPU packages</strong>
      ({total_entries} top-level classes &amp; functions + {total_methods} class methods).
      Type any keyword below to filter instantly, or click a package pill to jump right to it.
    </p>

    <input id="search" type="search" placeholder="Search functions, classes, or topics (e.g. cosine, bm25, rolling, conv2d, umap, haversine, groupby)..." autocomplete="off" />
    <div id="status" class="match-status"></div>

    <div class="pill-bar">{"".join(jump_pills)}</div>

    {"".join(pkg_blocks)}
  </div>

  <script>
    const input = document.getElementById('search');
    const status = document.getElementById('status');
    const sections = Array.from(document.querySelectorAll('.pkg-sec'));

    input.addEventListener('input', () => {{
      const q = input.value.trim().toLowerCase();
      let shown = 0;
      sections.forEach(sec => {{
        const items = Array.from(sec.querySelectorAll('.api-item'));
        let secHits = 0;
        items.forEach(item => {{
          const ok = !q || item.dataset.search.includes(q);
          item.style.display = ok ? '' : 'none';
          if (ok) {{ secHits++; shown++; }}
        }});
        sec.style.display = (!q || secHits > 0) ? '' : 'none';
      }});
      status.textContent = q ? `Showing ${{shown}} matching APIs for "${{input.value.trim()}}"` : '';
    }});
  </script>
</body>
</html>
"""
    out = Path("docs/reference.html")
    out.write_text(page, encoding="utf-8")
    print(f"Generated {out} -> 16 packages, {total_entries} top-level APIs, {total_methods} class methods")
    return total_entries, total_methods


if __name__ == "__main__":
    build_reference()
