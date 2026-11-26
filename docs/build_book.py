"""Rebuild docs/chapters/*.html, docs/glossary.html, & docs/index.html with SynthWave '84 theme + Acronym Glossary."""
from __future__ import annotations

import html as html_lib
import re
from pathlib import Path

CHAPTERS = [
    ("examples/00_ai_foundations/01_the_perceptron.ipynb",         "00_the_perceptron.html",            "Foundations &middot; The Perceptron"),
    ("examples/00_ai_foundations/02_multi_layer_perceptron.ipynb", "00b_multi_layer_perceptron.html",   "Foundations &middot; Multi-Layer Perceptrons"),
    ("examples/tldr_end2end/01_housing_classical_ml.ipynb",        "01_housing_classical_ml.html",      "Ch. 1 &middot; Housing Classical ML"),
    ("examples/tldr_end2end/02_housing_neural_network.ipynb",      "02_housing_neural_network.html",    "Ch. 2 &middot; Housing Neural Net"),
    ("examples/tldr_end2end/03_fashion_classical_ml.ipynb",        "03_fashion_classical_ml.html",      "Ch. 3 &middot; Fashion Classical CV"),
    ("examples/tldr_end2end/04_fashion_cnn_deep_learning.ipynb",   "04_fashion_cnn_deep_learning.html", "Ch. 4 &middot; Fashion 2D CNN"),
    ("examples/tldr_end2end/05_vector_search_classical_ml.ipynb",  "05_vector_search_classical_ml.html","Ch. 5 &middot; Vector Search &amp; Classical NLP"),
    ("examples/tldr_end2end/06_vector_embeddings_neural_net.ipynb","06_vector_embeddings_neural_net.html","Ch. 6 &middot; Neural Embeddings &amp; Deep NLP"),
]

# (Term, Full Name, Category, Plain-English Explanation)
GLOSSARY = [
    ("AI", "Artificial Intelligence", "General",
     "The broad field of building computer programs that can learn patterns, make decisions, or generate text and images."),
    ("ML", "Machine Learning", "General",
     "Teaching a computer to learn rules from data examples instead of hand-coding every if-statement yourself."),
    ("DL", "Deep Learning", "General",
     "A branch of Machine Learning that stacks multiple layers of artificial neurons (neural networks) to learn complex patterns."),
    ("NN", "Neural Network", "Deep Learning",
     "A network of interconnected artificial neurons organized in layers that learn weights from data via backpropagation."),
    ("MLP", "Multi-Layer Perceptron", "Deep Learning",
     "The classic neural network: stacked layers of neurons with non-linear activation functions (like GELU or ReLU) in between so it can learn curved boundaries."),
    ("CNN", "Convolutional Neural Network", "Deep Learning",
     "A neural network designed for images that slides small learned 2D filters (like 3x3 patches) across pixels to detect edges, shapes, and textures."),
    ("LLM", "Large Language Model", "Deep Learning",
     "A giant Transformer neural network (like Llama or GPT) trained on massive text datasets to predict the next word/token."),
    ("FFN", "Feed-Forward Network", "Deep Learning",
     "The MLP block inside every Transformer layer that processes each token independently and stores most of an LLM's factual knowledge."),
    ("GELU", "Gaussian Error Linear Unit", "Activations",
     "A smooth modern activation function used in Transformers and ResNets that gently bends negative values toward zero so stacked layers can learn curves."),
    ("ReLU", "Rectified Linear Unit", "Activations",
     "The simplest popular activation function: max(0, x). Keeps positive numbers unchanged and turns negative numbers into 0."),
    ("SiLU", "Sigmoid Linear Unit (Swish)", "Activations",
     "A smooth activation function defined as x * sigmoid(x), widely used inside modern Llama/Gemma SwiGLU blocks."),
    ("SwiGLU", "Swish-Gated Linear Unit", "Deep Learning",
     "A modern gated MLP block (silu(gate) * up) used in Llama and Gemma Transformers that outperforms standard ReLU/GELU FFNs."),
    ("LayerNorm", "Layer Normalization", "Deep Learning",
     "Rescales each row of neuron outputs to mean 0 and variance 1 so deep networks train smoothly without exploding or vanishing numbers."),
    ("RMSNorm", "Root Mean Square Normalization", "Deep Learning",
     "A simpler, faster version of LayerNorm (skips subtracting the mean) used in modern LLMs like Llama and Gemma."),
    ("SGD", "Stochastic Gradient Descent", "Training",
     "The classic training algorithm that nudges model weights downhill (against the gradient) on small random batches of data."),
    ("AdamW", "Adam with Decoupled Weight Decay", "Training",
     "The go-to modern neural network optimizer: adapts the learning rate for every weight individually and gently shrinks large weights to prevent overfitting."),
    ("MSE", "Mean Squared Error", "Metrics & Losses",
     "Average of (prediction - actual)^2. Penalizes big mistakes heavily; the standard training loss for regression."),
    ("RMSE", "Root Mean Squared Error", "Metrics & Losses",
     "Square root of MSE — puts your typical prediction error back into real-world units (like dollars for house prices!). Lower is better."),
    ("MAE", "Mean Absolute Error", "Metrics & Losses",
     "Average of |prediction - actual|. Your average mistake in real-world units, less sensitive to extreme outliers than RMSE."),
    ("BCE", "Binary Cross-Entropy", "Metrics & Losses",
     "The standard loss function for Yes/No (0 vs 1) classification that rewards confident correct probabilities and heavily penalizes confident wrong guesses."),
    ("EDA", "Exploratory Data Analysis", "Data & ML",
     "Inspecting and visualizing your dataset (maps, histograms, missing values, caps/outliers) BEFORE training any model."),
    ("PCA", "Principal Component Analysis", "Classical ML",
     "A classic dimensionality-reduction algorithm that compresses many correlated features (like 784 pixels) into a smaller set of top visual patterns."),
    ("KNN", "K-Nearest Neighbors", "Classical ML",
     "A simple non-linear algorithm that predicts an answer by finding the K most similar training examples and averaging or voting on their labels."),
    ("OvR", "One-vs-Rest", "Classical ML",
     "A strategy that turns a binary (Yes/No) classifier into a multi-class classifier by training one model per class ('Is it a Shirt? Is it a Sneaker? ...')."),
    ("CV", "Computer Vision", "General",
     "Teaching computers to understand images and video — from hand-crafted edge detectors (Classical CV) to learned 2D CNNs."),
    ("HOG", "Histogram of Oriented Gradients", "Classical ML",
     "A classic computer vision feature descriptor that summarizes the direction and strength of edges inside small image grid cells."),
    ("RBF", "Radial Basis Function", "Classical ML",
     "A smooth bell-curve similarity score exp(-dist^2 / 2*sigma^2) that equals 1 right next to a landmark/hub and fades smoothly to 0 far away."),
    ("CSV", "Comma-Separated Values", "Data & ML",
     "A plain-text table format where each line is a row and columns are separated by commas."),
    ("XOR", "Exclusive OR", "Logic Gates",
     "A logic rule that outputs 1 when the two inputs are DIFFERENT ((0,1) or (1,0)) and 0 when they are the same. Famous because 1 straight line cannot solve it!"),
    ("NAND", "Not-AND", "Logic Gates",
     "Outputs 0 ONLY when both inputs are 1 (and 1 otherwise). Every digital computer chip can be built purely from NAND gates!"),
    ("NOR", "Not-OR", "Logic Gates",
     "Outputs 1 ONLY when both inputs are 0 ('neither x1 NOR x2')."),
    ("GPU", "Graphics Processing Unit", "Hardware & Vulkan",
     "A chip with thousands of tiny parallel cores originally built for 3D graphics, now ideal for doing millions of AI math operations at once."),
    ("VRAM", "Video RAM (GPU Memory)", "Hardware & Vulkan",
     "The high-speed memory attached to your GPU where tensors, model weights, and shaders live while training."),
    ("APU", "Accelerated Processing Unit", "Hardware & Vulkan",
     "A single chip combining a CPU and GPU that share unified memory — like the AMD chip inside the Steam Deck."),
    ("CUDA", "Compute Unified Device Architecture", "Hardware & Vulkan",
     "NVIDIA's proprietary GPU programming platform. Most AI libraries require it — vkdata uses open Vulkan instead so you don't need an NVIDIA card!"),
    ("ROCm", "Radeon Open Compute", "Hardware & Vulkan",
     "AMD's GPU compute platform, which only supports select desktop/datacenter Radeon cards — unlike Vulkan, which works on almost any GPU."),
    ("SPIR-V", "Standard Portable Intermediate Representation", "Hardware & Vulkan",
     "The compact binary bytecode that Vulkan GPUs execute. Our fused kernels compile GLSL shaders into SPIR-V once and run natively in VRAM."),
    ("GLSL", "OpenGL Shading Language", "Hardware & Vulkan",
     "A C-like language for writing GPU compute shaders that compile to SPIR-V for Vulkan."),
    ("LUT", "Lookup Table", "Data & ML",
     "A fast 256-entry color table (used by vkcolorcet) that maps numbers to RGB colors in one instant lookup."),
    ("R²", "Coefficient of Determination (R-Squared)", "Metrics & Losses",
     "Measures what fraction of the variation in the target your model explains. 1.00 = perfect predictions, 0.00 = no better than guessing the average."),
    ("Conv2d", "2D Convolution Layer", "Deep Learning",
     "Slides small learned 2D filters (like 3x3 patches) across an image to detect edges, shapes, and textures anywhere in the picture."),
    ("MaxPool2d", "2D Max Pooling Layer", "Deep Learning",
     "Shrinks a 2D feature map by taking the strongest value in each small patch (e.g. 2x2), making a CNN less sensitive to tiny pixel shifts."),
    ("KMeans", "K-Means Clustering", "Classical ML",
     "An unsupervised algorithm that automatically groups points into K natural clusters (like discovering 12 California housing sub-markets from lat/lon!)."),
    ("Ridge", "Ridge Regression (L2 Regularization)", "Classical ML",
     "Linear regression with a gentle penalty (alpha) on huge weights so correlated features don't make the model overfit."),
    ("Dropout", "Dropout Regularization", "Training",
     "Randomly turns off a small fraction of neurons during training so the network can't memorize single examples and generalizes better."),
    ("Sigmoid", "Sigmoid Activation Function", "Activations",
     "Squashes any number from -infinity..+infinity into a smooth probability between 0.0 and 1.0 (an S-shaped curve)."),
    ("Softmax", "Softmax Function", "Activations",
     "Turns a list of raw class scores (logits) into positive probabilities that add up to 1.0 (100%)."),
    ("Epoch", "Training Epoch", "Training",
     "One full pass through the entire training dataset during model training."),
    ("Tensor", "Multi-Dimensional GPU Array", "Deep Learning",
     "A grid of numbers (1D vector, 2D matrix, 4D image batch) stored in GPU VRAM that tracks gradients for automatic backpropagation."),
    ("Autograd", "Automatic Differentiation", "Deep Learning",
     "The engine inside vktorch/PyTorch that remembers every math operation in the forward pass and automatically computes exact gradients on .backward()."),
    ("Backpropagation", "Backward Propagation of Errors", "Deep Learning",
     "Using the chain rule of calculus in reverse from the loss back to every weight so the optimizer knows which way to nudge each knob."),
    ("Overfitting", "Memorizing Training Data", "Training",
     "When a model learns noise in the training set instead of the true pattern, scoring great on training data but poorly on unseen test data."),
    ("RoPE", "Rotary Positional Embeddings", "Deep Learning",
     "Encodes each token's position by rotating pairs of Query/Key features in 2D space — the standard position encoding used in Llama, Gemma, and Qwen."),
    ("KV-Cache", "Key-Value Cache", "Deep Learning",
     "Saves past attention Keys and Values in GPU VRAM during text generation so each new token only computes 1 step instead of reprocessing the whole prompt."),
    ("LoRA", "Low-Rank Adaptation", "Training",
     "Fine-tunes a frozen pretrained model by adding two tiny trainable matrices (A and B) alongside a Linear layer — then merges them back with zero inference overhead."),
    ("TF-IDF", "Term Frequency - Inverse Document Frequency", "Vector & Text",
     "A classic formula that turns text into a vector by boosting distinctive topic words and down-weighting common filler words like 'the' and 'said'."),
    ("Embedding", "Dense Meaning Vector", "Vector & Text",
     "A learned list of numbers (e.g. 64 or 4,096 floats) representing a word, sentence, or image so that similar meanings live close together in vector space."),
    ("Cosine Similarity", "Angle-Based Vector Similarity", "Vector & Text",
     "Measures how closely two vectors point in the same direction (1.0 = identical direction, 0.0 = unrelated). For unit-length vectors, it's just one dot product!"),
    ("RAG", "Retrieval-Augmented Generation", "Vector & Text",
     "Searching a Vector Database for the most relevant documents first, then feeding those retrieved snippets into an LLM so it answers with grounded facts."),
    ("UMAP", "Uniform Manifold Approximation and Projection", "Vector & Text",
     "A fast dimensionality-reduction algorithm that projects high-dimensional vectors (like 64-D or 2,000-D embeddings) into 2D/3D maps while keeping clusters intact."),
    ("BoW", "Bag-of-Words", "Vector & Text",
     "The simplest text-to-vector method: count how many times each vocabulary word appears in a document, ignoring grammar and word order."),
    ("NLP", "Natural Language Processing", "Vector & Text",
     "The branch of AI focused on helping computers read, search, classify, and generate human language — from TF-IDF keyword vectors to neural embeddings and LLMs."),
    ("Tokenization", "Splitting Text into Tokens", "Vector & Text",
     "Step 1 of every NLP pipeline: chopping raw text into clean pieces (words, subwords, or characters) and mapping each piece to an integer ID so a computer can do math on it."),
    ("BM25", "Best Matching 25 (Okapi BM25)", "Vector & Text",
     "The gold-standard full-text keyword search algorithm used by Elasticsearch and Lucene — scores exact word matches while balancing term frequency and document length."),
    ("RRF", "Reciprocal Rank Fusion", "Vector & Text",
     "A simple, scale-invariant formula (sum of 1 / (60 + rank)) that combines ranked results from keyword search (BM25) and neural vector search into one stronger Hybrid Search list."),
    ("Hybrid Search", "BM25 Keyword + Neural Vector Search", "Vector & Text",
     "Running exact full-text search (BM25) and semantic GPU vector search together and fusing them with RRF — the production standard for RAG so both rare names/codes and synonyms match."),
]

def _slug(term: str) -> str:
    return "term-" + re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")

GLOSSARY_MAP = {term: (_slug(term), full, desc) for term, full, _, desc in GLOSSARY}
# Sort longest terms first so e.g. 'RMSNorm' matches before 'NN' or 'ML', 'RMSE' before 'MSE'
SORTED_TERMS = sorted(GLOSSARY_MAP.keys(), key=len, reverse=True)
TERM_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in SORTED_TERMS) + r")\b")

def link_acronyms_in_markdown(html: str) -> str:
    """Safely wrap acronyms inside .jp-RenderedMarkdown prose only (never inside code/pre/a/headings)."""
    def process_md_block(block_match: re.Match) -> str:
        block = block_match.group(0)
        # Tokenize HTML tags vs text
        parts = re.split(r"(<[^>]+>)", block)
        out = []
        skip_depth = 0
        skip_tags = {"code", "pre", "a", "h1", "h2", "h3", "h4", "script", "style"}
        for part in parts:
            if part.startswith("<"):
                m = re.match(r"^<\s*(/?)\s*([a-zA-Z0-9]+)", part)
                if m:
                    is_close, tag = bool(m.group(1)), m.group(2).lower()
                    if tag in skip_tags:
                        if is_close:
                            skip_depth = max(0, skip_depth - 1)
                        elif not part.rstrip().endswith("/>"):
                            skip_depth += 1
                out.append(part)
            else:
                if skip_depth == 0 and part.strip():
                    def repl(tm: re.Match) -> str:
                        t = tm.group(1)
                        slug, full, desc = GLOSSARY_MAP[t]
                        tip = html_lib.escape(f"{full}: {desc}", quote=True)
                        return (f'<a class="vk-term" href="../glossary.html#{slug}" title="{tip}" '
                                f'data-term="{html_lib.escape(t, quote=True)}" '
                                f'data-full="{html_lib.escape(full, quote=True)}" '
                                f'data-desc="{html_lib.escape(desc, quote=True)}">{t}</a>')
                    out.append(TERM_RE.sub(repl, part))
                else:
                    out.append(part)
        res = "".join(out)
        # Also make standalone inline <code>TERM</code> (and <code>nn.TERM()</code>) in markdown prose tap-explainable!
        def code_repl(cm: re.Match) -> str:
            inner = cm.group(1)
            clean = inner.strip().rstrip("()").split(".")[-1]
            if clean in GLOSSARY_MAP:
                slug, full, desc = GLOSSARY_MAP[clean]
                tip = html_lib.escape(f"{full}: {desc}", quote=True)
                return (f'<a class="vk-term vk-code-term" href="../glossary.html#{slug}" title="{tip}" '
                        f'data-term="{html_lib.escape(clean, quote=True)}" '
                        f'data-full="{html_lib.escape(full, quote=True)}" '
                        f'data-desc="{html_lib.escape(desc, quote=True)}"><code>{inner}</code></a>')
            return cm.group(0)
        # Only replace <code>...</code> that isn't already inside an <a> or <pre>
        res = re.sub(r'(?<!<pre>)<code>([^<]{1,40})</code>', code_repl, res)
        return res

    # Match each RenderedMarkdown div up to its closing wrapper (nbconvert emits clean single-level divs inside)
    return re.sub(
        r'<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput[^"]*"[^>]*>.*?</div>\s*</div>\s*</div>',
        process_md_block,
        html,
        flags=re.S,
    )


SYNTHWAVE84_CSS = """
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --jp-layout-color0: #262335 !important;
    --jp-layout-color1: #1e1a2b !important;
    --jp-layout-color2: #2a2139 !important;
    --jp-border-color1: #495495 !important;
    --jp-border-color2: #3b3355 !important;
    --jp-ui-font-color1: #f4eee4 !important;
    --jp-content-font-color1: #f4eee4 !important;
    --jp-cell-editor-background: #1b1728 !important;
    --jp-cell-editor-border-color: #495495 !important;
    --jp-mirror-editor-keyword-color: #ff7edb !important;
    --jp-mirror-editor-atom-color: #f97e72 !important;
    --jp-mirror-editor-number-color: #fede5d !important;
    --jp-mirror-editor-def-color: #72f1b8 !important;
    --jp-mirror-editor-variable-color: #f4eee4 !important;
    --jp-mirror-editor-variable-2-color: #ff7edb !important;
    --jp-mirror-editor-variable-3-color: #72f1b8 !important;
    --jp-mirror-editor-punctuation-color: #b6b1d6 !important;
    --jp-mirror-editor-property-color: #36f9f6 !important;
    --jp-mirror-editor-operator-color: #ff7edb !important;
    --jp-mirror-editor-comment-color: #848bbd !important;
    --jp-mirror-editor-string-color: #ff9e64 !important;
    --jp-mirror-editor-string-2-color: #ff9e64 !important;
    --jp-mirror-editor-meta-color: #72f1b8 !important;
    --jp-mirror-editor-qualifier-color: #ff7edb !important;
    --jp-mirror-editor-builtin-color: #36f9f6 !important;
    --jp-mirror-editor-bracket-color: #b6b1d6 !important;
    --jp-mirror-editor-tag-color: #ff7edb !important;
    --jp-mirror-editor-attribute-color: #36f9f6 !important;
    --jp-mirror-editor-header-color: #ff7edb !important;
    --jp-mirror-editor-quote-color: #72f1b8 !important;
    --jp-mirror-editor-link-color: #36f9f6 !important;
    --jp-mirror-editor-error-color: #fe4450 !important;
  }
  html, body, body.jp-Notebook {
    background-color: #262335 !important;
    background-image: radial-gradient(circle at 20% 0%, rgba(255, 126, 219, 0.09), transparent 45%),
                      radial-gradient(circle at 85% 10%, rgba(54, 249, 246, 0.08), transparent 40%) !important;
    color: #f4eee4 !important;
  }
  .jp-RenderedMarkdown {
    color: #ede6f5 !important;
    line-height: 1.68;
  }
  .jp-RenderedMarkdown h1 {
    color: #ff7edb !important;
    text-shadow: 0 0 12px rgba(255, 126, 219, 0.45);
    border-color: #495495 !important;
  }
  .jp-RenderedMarkdown h2 {
    color: #36f9f6 !important;
    text-shadow: 0 0 10px rgba(54, 249, 246, 0.35);
    border-color: #495495 !important;
  }
  .jp-RenderedMarkdown h3 {
    color: #fede5d !important;
    text-shadow: 0 0 8px rgba(254, 222, 93, 0.3);
  }
  .jp-RenderedMarkdown strong { color: #ffffff !important; }
  .jp-RenderedMarkdown a { color: #36f9f6 !important; }
  .jp-RenderedMarkdown a.vk-term {
    color: inherit !important;
    text-decoration: underline dotted #36f9f6 !important;
    text-underline-offset: 3px;
    cursor: help;
  }
  .jp-RenderedMarkdown a.vk-code-term code { border-bottom: 1.5px dotted #36f9f6 !important; }
  .jp-RenderedMarkdown a.vk-term:hover {
    color: #36f9f6 !important;
    text-shadow: 0 0 8px rgba(54, 249, 246, 0.45);
  }
  .jp-RenderedMarkdown code {
    background: #1b1728 !important; color: #ff7edb !important;
    border: 1px solid rgba(255, 126, 219, 0.25);
    padding: 2px 6px; border-radius: 5px;
  }
  .jp-RenderedMarkdown blockquote {
    border-left: 3px solid #ff7edb !important;
    background: #1e1a2b; padding: 10px 14px; border-radius: 0 8px 8px 0;
    color: #d6cceb !important;
    box-shadow: 0 0 16px rgba(255, 126, 219, 0.08);
  }
  .jp-RenderedMarkdown hr {
    border: none; height: 1px;
    background: linear-gradient(90deg, rgba(255,126,219,0.6), rgba(54,249,246,0.6), transparent);
  }
  .jp-InputArea-editor {
    background: #1b1728 !important;
    border: 1px solid #495495 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 18px rgba(54, 249, 246, 0.07);
  }
  .jp-OutputArea-output pre {
    color: #d7cef0 !important;
    background: #171422 !important;
    border: 1px solid rgba(73, 84, 149, 0.45);
    border-radius: 8px; padding: 10px 12px;
  }

  /* SynthWave '84 Neon Vaporwave Python Syntax Highlighting (harmonized, zero harsh red) */
  .jp-CodeMirrorEditor, .jp-InputArea-editor, .highlight, .highlight pre { background: #1b1728 !important; color: #f4eee4 !important; }
  .highlight .c, .highlight .c1, .highlight .cm, .highlight .ch { color: #848bbd !important; font-style: italic; }
  .highlight .k, .highlight .kn, .highlight .kd, .highlight .kc, .highlight .ow { color: #ff7edb !important; font-weight: 600; }
  .highlight .o { color: #ff7edb !important; }
  .highlight .nn, .highlight .nc, .highlight .nb, .highlight .bp { color: #36f9f6 !important; font-weight: 500; }
  .highlight .nf, .highlight .fm, .highlight .nd { color: #72f1b8 !important; font-weight: 600; }
  .highlight .s, .highlight .s1, .highlight .s2, .highlight .sa, .highlight .sb, .highlight .sc, .highlight .sd, .highlight .se, .highlight .si { color: #ff9e64 !important; }
  .highlight .m, .highlight .mi, .highlight .mf, .highlight .mh, .highlight .mo { color: #fede5d !important; }
  .highlight .n { color: #f4eee4 !important; }
  .highlight .p { color: #b6b1d6 !important; }

  /* Sticky SynthWave '84 top & bottom chapter navigation */
  .vkbook-nav {
    position: sticky; top: 0; z-index: 1000;
    background: linear-gradient(90deg, #1b1528, #241b35, #1b1528);
    color: #f4eee4;
    border-bottom: 1px solid #ff7edb;
    box-shadow: 0 2px 16px rgba(255, 126, 219, 0.22);
    padding: 10px 14px; display: flex; flex-wrap: wrap;
    align-items: center; justify-content: space-between; gap: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
  }
  .vkbook-nav a {
    color: #36f9f6; text-decoration: none; font-weight: 600;
    padding: 5px 10px; border-radius: 6px; background: #2a2139;
    border: 1px solid rgba(54, 249, 246, 0.45);
    text-shadow: 0 0 6px rgba(54, 249, 246, 0.35);
  }
  .vkbook-nav a.gloss-btn {
    color: #fede5d; border-color: rgba(254, 222, 93, 0.45);
    text-shadow: 0 0 6px rgba(254, 222, 93, 0.3);
  }
  .vkbook-nav a:hover { background: #382c4f; border-color: #ff7edb; color: #ff7edb; }
  .vkbook-nav .title {
    color: #ff7edb; font-weight: 700;
    text-shadow: 0 0 10px rgba(255, 126, 219, 0.45);
  }
  .vkbook-footer {
    margin: 32px 0 24px; padding: 14px;
    border-top: 1px solid #495495;
    display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .vkbook-footer a {
    color: #36f9f6; text-decoration: none; font-weight: 600;
    padding: 8px 14px; background: #1e1a2b;
    border: 1px solid rgba(54, 249, 246, 0.45); border-radius: 8px;
  }

  /* In-chapter SynthWave '84 tap-to-explain glossary drawer (never lose your place!) */
  #vk-tip-box {
    position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%);
    width: min(94vw, 540px); z-index: 2000; display: none;
    background: #1b1528; color: #f4eee4;
    border: 1px solid #36f9f6; border-radius: 14px; padding: 14px 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.65), 0 0 22px rgba(54, 249, 246, 0.28);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  #vk-tip-box .vk-tip-head {
    display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  }
  #vk-tip-box .vk-tip-term { color: #fede5d; font-weight: 800; font-size: 18px; }
  #vk-tip-box .vk-tip-full { color: #36f9f6; font-weight: 600; font-size: 14px; margin: 2px 0 6px; }
  #vk-tip-box .vk-tip-desc { color: #ede6f5; font-size: 14px; margin: 0 0 10px; line-height: 1.5; }
  #vk-tip-box .vk-tip-actions { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  #vk-tip-box button {
    background: #2a2139; color: #ff7edb; border: 1px solid rgba(255,126,219,0.5);
    border-radius: 7px; padding: 5px 12px; font-weight: 700; cursor: pointer; font-size: 13px;
  }
  #vk-tip-box a { color: #36f9f6; font-size: 13px; font-weight: 600; text-decoration: none; }

  @media (max-width: 768px) {
    body.jp-Notebook { padding: 0 !important; margin: 0 !important; }
    .jp-Notebook { padding: 8px !important; }
    .jp-Cell { padding: 4px 2px !important; }
    .jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
    .jp-RenderedImage img, img { max-width: 100% !important; height: auto !important; }
    pre, code { font-size: 12.5px !important; }
    .jp-CodeMirrorEditor, .jp-OutputArea-output pre { overflow-x: auto !important; }
    table { display: block; overflow-x: auto; max-width: 100%; }
  }
</style>
"""

POPOVER_JS = """
<div id="vk-tip-box" role="dialog" aria-live="polite">
  <div class="vk-tip-head">
    <span class="vk-tip-term" id="vk-t-abbr"></span>
    <button type="button" onclick="document.getElementById('vk-tip-box').style.display='none'">Close</button>
  </div>
  <div class="vk-tip-full" id="vk-t-full"></div>
  <p class="vk-tip-desc" id="vk-t-desc"></p>
  <div class="vk-tip-actions">
    <span style="font-size:12px;color:#848bbd;">Tap anywhere outside to dismiss</span>
    <a id="vk-t-more" href="../glossary.html">Open Full Glossary &rarr;</a>
  </div>
</div>
<script>
  document.addEventListener('click', (e) => {
    const box = document.getElementById('vk-tip-box');
    const a = e.target.closest('a.vk-term');
    if (a) {
      e.preventDefault();
      document.getElementById('vk-t-abbr').textContent = a.dataset.term || a.textContent;
      document.getElementById('vk-t-full').textContent = a.dataset.full || '';
      document.getElementById('vk-t-desc').textContent = a.dataset.desc || a.title || '';
      document.getElementById('vk-t-more').href = a.getAttribute('href');
      box.style.display = 'block';
    } else if (box.style.display === 'block' && !e.target.closest('#vk-tip-box')) {
      box.style.display = 'none';
    }
  });
</script>
"""

chap_dir = Path("docs/chapters")
for idx, (_, fname, title) in enumerate(CHAPTERS):
    fp = chap_dir / fname
    html = fp.read_text(encoding="utf-8")
    if "vkbook-nav" in html:
        # Ensure existing styled chapters also get the API Docs button
        if "../reference.html" not in html:
            html = html.replace(
                '<a class="gloss-btn" href="../glossary.html">Glossary</a>',
                '<a class="gloss-btn" href="../glossary.html">Glossary</a><a class="gloss-btn" href="../reference.html">API Docs</a>',
                1,
            )
            html = html.replace(
                '<a href="../glossary.html">Acronym Glossary</a>',
                '<a href="../glossary.html">Acronym Glossary</a><a href="../reference.html">vkdata API Docs</a>',
                1,
            )
            fp.write_text(html, encoding="utf-8")
            print("Added API Docs link ->", fp)
        continue
    html = link_acronyms_in_markdown(html)
    prev_link = f'<a href="{CHAPTERS[idx-1][1]}">&larr; Prev</a>' if idx > 0 else ""
    next_link = f'<a href="{CHAPTERS[idx+1][1]}">Next &rarr;</a>' if idx + 1 < len(CHAPTERS) else ""
    nav_html = (
        f'<div class="vkbook-nav">'
        f'<span style="display:flex;gap:6px;align-items:center;">'
        f'<a href="../index.html">&larr; Home</a>'
        f'<a class="gloss-btn" href="../glossary.html">Glossary</a>'
        f'<a class="gloss-btn" href="../reference.html">API Docs</a>'
        f'</span>'
        f'<span class="title">{title}</span>'
        f'<span style="display:flex;gap:6px;">{prev_link}{next_link}</span>'
        f'</div>'
    )
    footer_html = (
        f'<div class="vkbook-footer">'
        f'{prev_link or "<a href=\"../index.html\">&larr; Book Home</a>"}'
        f'<a href="../glossary.html">Acronym Glossary</a>'
        f'<a href="../reference.html">vkdata API Docs</a>'
        f'{next_link or "<a href=\"../index.html\">Back to Table of Contents &rarr;</a>"}'
        f'</div>'
    )
    html = html.replace("</head>", SYNTHWAVE84_CSS + "\n</head>", 1)
    bi = html.find(">", html.find("<body")) + 1
    html = html[:bi] + "\n" + nav_html + "\n" + html[bi:]
    html = html.replace("</body>", "\n" + footer_html + POPOVER_JS + "\n</body>", 1)
    fp.write_text(html, encoding="utf-8")
    print("Styled SynthWave '84 + Glossary links ->", fp)

# Generate searchable SynthWave '84 docs/glossary.html
cards_html = []
for term, full, cat, desc in sorted(GLOSSARY, key=lambda r: r[0].lower()):
    slug = _slug(term)
    cards_html.append(f"""      <div class="term-card" id="{slug}" data-search="{html_lib.escape((term + ' ' + full + ' ' + cat + ' ' + desc).lower())}">
        <div class="term-head">
          <span class="term-abbr">{html_lib.escape(term)}</span>
          <span class="term-cat">{html_lib.escape(cat)}</span>
        </div>
        <div class="term-full">{html_lib.escape(full)}</div>
        <p class="term-desc">{html_lib.escape(desc)}</p>
      </div>""")

glossary_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Acronym Glossary &amp; Index &mdash; Hands-On AI, ML &amp; GPU Computing (vkdata)</title>
  <style>
    :root {{
      --bg: #262335; --card: #1e1a2b; --border: #495495;
      --text: #f4eee4; --muted: #b6b1d6;
      --pink: #ff7edb; --cyan: #36f9f6; --yellow: #fede5d; --mint: #72f1b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background-color: var(--bg); color: var(--text);
      background-image:
        radial-gradient(circle at 18% 0%, rgba(255, 126, 219, 0.14), transparent 42%),
        radial-gradient(circle at 85% 5%, rgba(54, 249, 246, 0.12), transparent 38%),
        linear-gradient(180deg, #241b2f 0%, #262335 45%, #1c1828 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.6; min-height: 100vh;
    }}
    .nav {{
      position: sticky; top: 0; z-index: 100;
      background: linear-gradient(90deg, #1b1528, #241b35, #1b1528);
      border-bottom: 1px solid var(--pink);
      box-shadow: 0 2px 16px rgba(255, 126, 219, 0.22);
      padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; gap: 10px;
    }}
    .nav a, .nav button {{
      color: var(--cyan); background: #2a2139; border: 1px solid rgba(54, 249, 246, 0.45);
      border-radius: 6px; padding: 5px 11px; font-weight: 600; font-size: 14px;
      text-decoration: none; cursor: pointer;
    }}
    .nav .title {{ color: var(--pink); font-weight: 700; text-shadow: 0 0 10px rgba(255, 126, 219, 0.45); }}
    .wrap {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 60px; }}
    h1 {{ color: var(--pink); text-shadow: 0 0 14px rgba(255, 126, 219, 0.5); margin: 0 0 8px; }}
    .sub {{ color: var(--muted); margin: 0 0 18px; font-size: 15px; }}
    .search {{
      width: 100%; padding: 11px 14px; border-radius: 10px;
      background: #1b1728; color: var(--text); font-size: 15px;
      border: 1px solid var(--cyan); outline: none; margin-bottom: 20px;
      box-shadow: 0 0 14px rgba(54, 249, 246, 0.12);
    }}
    .search::placeholder {{ color: #848bbd; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .term-card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px 16px; scroll-margin-top: 72px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .term-card:target {{
      border-color: var(--cyan);
      box-shadow: 0 0 24px rgba(54, 249, 246, 0.35);
      background: #231d36;
    }}
    .term-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }}
    .term-abbr {{
      font-size: 20px; font-weight: 800; color: var(--yellow);
      text-shadow: 0 0 8px rgba(254, 222, 93, 0.35);
    }}
    .term-cat {{
      font-size: 12px; font-weight: 700; color: var(--mint);
      background: rgba(114, 241, 184, 0.12); border: 1px solid rgba(114, 241, 184, 0.35);
      padding: 2px 8px; border-radius: 6px;
    }}
    .term-full {{ color: var(--cyan); font-weight: 600; font-size: 14.5px; margin: 2px 0 6px; }}
    .term-desc {{ color: var(--text); margin: 0; font-size: 14.5px; }}
  </style>
</head>
<body>
  <div class="nav">
    <button onclick="if (history.length > 1) history.back(); else location.href='index.html';">&larr; Back</button>
    <span class="title">Acronym Glossary ({len(GLOSSARY)} Terms)</span>
    <span style="display:flex;gap:6px;flex-wrap:wrap;">
      <a href="reference.html" style="color:var(--mint);border-color:rgba(114,241,184,0.45);">&#128218; API Docs</a>
      <a href="index.html">Book Home</a>
    </span>
  </div>
  <div class="wrap">
    <h1>Plain-English Acronym Glossary</h1>
    <p class="sub">Every acronym in the book is clickable (dotted cyan underline) and links straight here. Looking for Python functions &amp; signatures across all 16 packages? Jump to the <a href="reference.html" style="color:var(--mint);font-weight:700;text-decoration:none;border-bottom:1px dotted var(--mint);">vkdata API Reference &rarr;</a></p>
    <input class="search" id="q" type="search" placeholder="Filter acronyms (e.g. MLP, RMSE, SPIR-V, GELU, PCA)..." autofocus>
    <div class="grid" id="list">
{chr(10).join(cards_html)}
    </div>
  </div>
  <script>
    const q = document.getElementById('q');
    const cards = Array.from(document.querySelectorAll('.term-card'));
    q.addEventListener('input', () => {{
      const v = q.value.trim().toLowerCase();
      for (const c of cards) c.style.display = (!v || c.dataset.search.includes(v)) ? '' : 'none';
    }});
  </script>
</body>
</html>
"""
Path("docs/glossary.html").write_text(glossary_html, encoding="utf-8")
print(f"Generated docs/glossary.html ({len(GLOSSARY)} terms)")

# Regenerate docs/index.html with prominent Glossary button/card
index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hands-On AI, ML &amp; GPU Computing — Without CUDA or ROCm (vkdata)</title>
  <style>
    :root {{
      --bg: #262335; --card: #1e1a2b; --border: #495495;
      --text: #f4eee4; --muted: #b6b1d6;
      --pink: #ff7edb; --cyan: #36f9f6; --yellow: #fede5d; --mint: #72f1b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background-color: var(--bg); color: var(--text);
      background-image:
        radial-gradient(circle at 18% 0%, rgba(255, 126, 219, 0.16), transparent 42%),
        radial-gradient(circle at 85% 5%, rgba(54, 249, 246, 0.14), transparent 38%),
        linear-gradient(180deg, #241b2f 0%, #262335 45%, #1c1828 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.6; min-height: 100vh;
    }}
    .wrap {{ max-width: 860px; margin: 0 auto; padding: 28px 18px 60px; }}
    .top-bar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 14px; }}
    .badge {{
      display: inline-block; font-size: 12px; font-weight: 700;
      color: var(--cyan); border: 1px solid rgba(54, 249, 246, 0.55);
      background: rgba(54, 249, 246, 0.1); padding: 4px 12px; border-radius: 999px;
      text-shadow: 0 0 8px rgba(54, 249, 246, 0.45);
    }}
    .gloss-pill {{
      display: inline-block; font-size: 12px; font-weight: 700; text-decoration: none;
      color: var(--yellow); border: 1px solid rgba(254, 222, 93, 0.55);
      background: rgba(254, 222, 93, 0.1); padding: 4px 12px; border-radius: 999px;
    }}
    .gloss-pill:hover {{ background: rgba(254, 222, 93, 0.22); }}
    h1 {{
      font-size: clamp(26px, 5vw, 40px); line-height: 1.18; margin: 0 0 12px;
      color: var(--pink); text-shadow: 0 0 18px rgba(255, 126, 219, 0.55);
    }}
    .sub {{ color: var(--muted); font-size: 16px; margin: 0 0 26px; }}
    h2 {{
      font-size: 20px; margin: 32px 0 14px; color: var(--cyan);
      text-shadow: 0 0 10px rgba(54, 249, 246, 0.4);
      border-bottom: 1px solid rgba(255, 126, 219, 0.35); padding-bottom: 6px;
    }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .dt-legend {{
      display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 14px;
    }}
    .dt-chip {{
      display: inline-flex; align-items: center; gap: 6px;
      font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 999px;
      background: rgba(255, 255, 255, 0.04); border: 1px solid var(--accent, var(--border));
      color: var(--accent, var(--text));
    }}
    .dt-chip .dot {{
      width: 8px; height: 8px; border-radius: 50%; background: var(--accent, var(--cyan));
      box-shadow: 0 0 8px var(--accent, var(--cyan));
    }}
    .card {{
      --accent: var(--pink);
      --accent-bg: rgba(255, 126, 219, 0.10);
      --accent-glow: rgba(255, 126, 219, 0.28);
      display: block; text-decoration: none; color: inherit;
      background: var(--card);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      border-radius: 12px; padding: 16px 18px;
      transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
    }}
    .card.dt-foundations {{ --accent: #fede5d; --accent-bg: rgba(254, 222, 93, 0.12);  --accent-glow: rgba(254, 222, 93, 0.25); }}
    .card.dt-tabular     {{ --accent: #36f9f6; --accent-bg: rgba(54, 249, 246, 0.11);  --accent-glow: rgba(54, 249, 246, 0.26); }}
    .card.dt-image       {{ --accent: #ff7edb; --accent-bg: rgba(255, 126, 219, 0.11); --accent-glow: rgba(255, 126, 219, 0.28); }}
    .card.dt-vector      {{ --accent: #72f1b8; --accent-bg: rgba(114, 241, 184, 0.11); --accent-glow: rgba(114, 241, 184, 0.26); }}
    .card.dt-timeseries  {{ --accent: #ff9e64; --accent-bg: rgba(255, 158, 100, 0.12); --accent-glow: rgba(255, 158, 100, 0.26); }}
    .card:hover {{
      border-color: var(--accent); transform: translateY(-2px);
      box-shadow: 0 0 22px var(--accent-glow);
    }}
    .card-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
    .ch {{
      color: var(--accent); font-weight: 700; font-size: 13px;
      text-transform: uppercase; letter-spacing: 0.05em;
      text-shadow: 0 0 8px var(--accent-glow);
    }}
    .pill {{
      font-size: 12px; font-weight: 700; color: var(--accent);
      background: var(--accent-bg); border: 1px solid var(--accent);
      padding: 2px 9px; border-radius: 6px;
    }}
    .card h3 {{ margin: 6px 0 6px; font-size: 18px; color: #ffffff; }}
    .card p {{ margin: 0; color: var(--muted); font-size: 14.5px; }}
    .note {{
      margin-top: 30px; background: #1a1626; border: 1px solid rgba(54, 249, 246, 0.4);
      border-radius: 12px; padding: 15px 18px; font-size: 14px; color: var(--muted);
    }}
    .note strong {{ color: var(--yellow); }}
    .note a {{ color: var(--cyan); text-decoration: none; font-weight: 600; }}
    code {{ background: #2b2340; padding: 2px 6px; border-radius: 5px; font-size: 13px; color: var(--pink); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top-bar">
      <span class="badge">Interactive Follow-Along Book &middot; Pure Vulkan GPU</span>
      <a class="gloss-pill" href="glossary.html">&#128214; Acronym Glossary ({len(GLOSSARY)} Terms)</a>
      <a class="gloss-pill" href="reference.html" style="color:var(--mint);border-color:rgba(114,241,184,0.55);background:rgba(114,241,184,0.10);">&#128218; vkdata API Reference (16 Packages)</a>
    </div>
    <h1>Hands-On AI, Machine Learning &amp; GPU Computing &mdash; Without CUDA or ROCm</h1>
    <p class="sub">
      Learn how AI actually works on the Vulkan GPU you already own (Steam Deck, AMD APU, Intel iGPU/Arc, older Radeon/NVIDIA, or Apple MoltenVK) &mdash; powered by the bundled <code>vkdata</code> toolkit.
      <em>Tip: Tap or hover any <span style="border-bottom:1px dotted #36f9f6;color:#f4eee4;">dotted acronym</span> in a chapter to jump straight to its plain-English definition!</em>
    </p>

    <h2>Part 0 &middot; AI Foundations: Building Blocks of Neural Networks</h2>
    <div class="grid">
      <a class="card dt-foundations" href="chapters/00_the_perceptron.html">
        <div class="card-head">
          <span class="ch">Foundations 1 &middot; 1 Artificial Neuron</span>
          <span class="pill">5 Gates &rarr; GPU Neuron &rarr; XOR Wall</span>
        </div>
        <h3>The Perceptron: How a Single Artificial Neuron Works</h3>
        <p>Start from zero: weighted votes (AND, OR, NOT, NAND, NOR), watching a single straight line learn with <code>vklearn.Perceptron</code>, running the same neuron on your Vulkan GPU with <code>vktorch</code>, and hitting the famous XOR limit.</p>
      </a>

      <a class="card dt-foundations" href="chapters/00b_multi_layer_perceptron.html">
        <div class="card-head">
          <span class="ch">Foundations 2 &middot; Hidden Layers &amp; LLMs</span>
          <span class="pill">XOR 100% &middot; 3 Params &rarr; 8B Llama</span>
        </div>
        <h3>Multi-Layer Perceptrons (MLPs): Bending Space, Solving XOR &amp; LLM Parameters</h3>
        <p>Wire 3 perceptrons together to solve XOR by hand, see why <code>GELU</code>/<code>ReLU</code> activations are mandatory, watch 4 neurons draw a diamond while 32&times;32 sculpts a smooth circle, and trace the parameter ladder up to 8B LLMs.</p>
      </a>
    </div>

    <h2>Part I &middot; End-to-End Projects: Tabular, Image &amp; Vector Data</h2>
    <p style="color:var(--muted);font-size:14px;margin:-6px 0 10px;">Every data type follows the same pattern: <strong>Feature Discovery &amp; Classical ML</strong> first, then <strong>Deep Learning (<code>vktorch</code>)</strong> on the exact same split.</p>
    <div class="dt-legend">
      <span class="dt-chip" style="--accent:#36f9f6;"><span class="dot"></span>Tabular Data (Ch. 1&ndash;2)</span>
      <span class="dt-chip" style="--accent:#ff7edb;"><span class="dot"></span>Image Data (Ch. 3&ndash;4)</span>
      <span class="dt-chip" style="--accent:#72f1b8;"><span class="dot"></span>Vector / Text &amp; NLP (Ch. 5&ndash;6)</span>
    </div>
    <div class="grid">
      <a class="card dt-tabular" href="chapters/01_housing_classical_ml.html">
        <div class="card-head">
          <span class="ch">Chapter 1 &middot; Tabular Data &middot; Regression</span>
          <span class="pill">$69.6k &rarr; $52.7k RMSE</span>
        </div>
        <h3>California Housing: Feature Discovery &amp; Classical ML</h3>
        <p>Load 20,640 Census districts, spot the $500k cap on a GPU map, engineer 36 domain + Haversine + KMeans-RBF features, and race LinearRegression, Ridge, and GPU KNN.</p>
      </a>

      <a class="card dt-tabular" href="chapters/02_housing_neural_network.html">
        <div class="card-head">
          <span class="ch">Chapter 2 &middot; Tabular Data &middot; Deep Learning</span>
          <span class="pill">$48.2k RMSE &middot; R&sup2; 0.828</span>
        </div>
        <h3>California Housing: Beating Classical ML with a Deep Neural Network</h3>
        <p>Train a <code>vktorch</code> Pre-Norm Residual MLP (fused SPIR-V LayerNorm, GELU, AdamW) on the exact same split &mdash; beating GPU KNN by -$4,510 RMSE with .npz checkpointing.</p>
      </a>

      <a class="card dt-image" href="chapters/03_fashion_classical_ml.html">
        <div class="card-head">
          <span class="ch">Chapter 3 &middot; Image Data &middot; Classical CV</span>
          <span class="pill">81.5% &rarr; 86.45% Acc</span>
        </div>
        <h3>Fashion-MNIST: Image Feature Discovery &amp; Classical Computer Vision</h3>
        <p>Visualize GPU &ldquo;Average Clothing&rdquo; prototypes, discover Eigen-Clothes with PCA, hand-craft Sobel/HOG edge features, and analyze the 10&times;10 confusion matrix.</p>
      </a>

      <a class="card dt-image" href="chapters/04_fashion_cnn_deep_learning.html">
        <div class="card-head">
          <span class="ch">Chapter 4 &middot; Image Data &middot; 2D CNN</span>
          <span class="pill">87.70% Acc &middot; 24.5s VRAM SPIR-V</span>
        </div>
        <h3>Fashion-MNIST: Beating Classical CV with a 2D CNN (<code>vktorch</code>)</h3>
        <p>Train a strided <code>Conv2d</code> CNN directly on raw 28&times;28 pixels in 24.5s on Steam Deck GPU (12.8&times; faster SPIR-V <code>im2col</code>), inspect 16 learned 3&times;3 filters, and beat every Classical CV model.</p>
      </a>

      <a class="card dt-vector" href="chapters/05_vector_search_classical_ml.html">
        <div class="card-head">
          <span class="ch">Chapter 5 &middot; Vector / Text Data &middot; Classical NLP</span>
          <span class="pill">vkvec GPU Search &middot; 97.30% Acc</span>
        </div>
        <h3>Full-Length BBC News: TF-IDF Vectors, <code>vkvec</code> RAG Chunking &amp; Classical NLP</h3>
        <p>Walk through the 3-step NLP pipeline on 2,225 complete multi-paragraph BBC articles, chunk long articles with <code>vkvec.chunk_text</code>, run filtered GPU Cosine Search, map 5 topic islands with <code>vkvis.UMAP</code>, and hit the Synonym Wall.</p>
      </a>

      <a class="card dt-vector" href="chapters/06_vector_embeddings_neural_net.html">
        <div class="card-head">
          <span class="ch">Chapter 6 &middot; Vector / Text Data &middot; Deep NLP &amp; Hybrid Search</span>
          <span class="pill">97.53% Acc &middot; 64-D Neural + BM25 RRF</span>
        </div>
        <h3>Full-Length BBC News: Learned Neural Embeddings &amp; <code>vkvec</code> Hybrid Search</h3>
        <p>Train a <code>vktorch</code> Neural Text Encoder in 12.2s on Steam Deck GPU to compress full articles into 64-D vectors (31&times; smaller), shatter the Synonym Wall (<code>0.000 &rarr; +0.999</code>), and fuse BM25 + Neural Search with <code>vkvec.HybridIndex</code>.</p>
      </a>
    </div>

    <h2>Reference &middot; Glossary &amp; <code>vkdata</code> Toolset Documentation</h2>
    <div class="grid">
      <a class="card dt-vector" href="reference.html">
        <div class="card-head">
          <span class="ch">API Reference &middot; 16 Vulkan GPU Packages</span>
          <span class="pill">Searchable Functions, Classes &amp; Signatures</span>
        </div>
        <h3><code>vkdata</code> Toolset Documentation (Complete Function Guide)</h3>
        <p>Browse and search every exported class, method, and function across <code>vknumpy</code>, <code>vkpandas</code>, <code>vkscipy</code>, <code>vklearn</code>, <code>vktorch</code>, <code>vkvec</code>, <code>vktime</code>, <code>vkvis</code>, <code>vkspatial</code>, <code>vkastro</code>, and the visualization stack &mdash; with signatures, plain-English explanations, and quick-start examples.</p>
      </a>

      <a class="card dt-foundations" href="glossary.html">
        <div class="card-head">
          <span class="ch">Glossary &middot; {len(GLOSSARY)} Plain-English Terms</span>
          <span class="pill">Tap-to-Explain in Every Chapter</span>
        </div>
        <h3>Plain-English AI, ML &amp; Vulkan Acronym Glossary</h3>
        <p>Instant definitions for every acronym and layer name in the book &mdash; from <code>MLP</code>, <code>GELU</code>, and <code>Conv2d</code> to <code>SPIR-V</code>, <code>TF-IDF</code>, <code>BM25</code>, <code>RRF</code>, and <code>RoPE</code>.</p>
      </a>
    </div>

    <div class="note">
      <strong>Want to run the notebooks interactively on your own GPU?</strong><br>
      Clone <a href="https://github.com/billyblackburn/vkdata">github.com/billyblackburn/vkdata</a> and run
      <code>pixi run setup &amp;&amp; pixi run build &amp;&amp; pixi run lab</code>.
      Parts II&ndash;VI (Vulkan GPU Puzzles, GPU DataFrames, UMAP/Manifold Learning, Classical ML at scale, and Mini-Llama) are all included in the repo!
    </div>
  </div>
</body>
</html>
"""
Path("docs/index.html").write_text(index_html, encoding="utf-8")
print("Styled SynthWave '84 -> docs/index.html")

# Also regenerate the searchable 16-package vkdata API reference
import sys
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reference import build_reference
build_reference()
