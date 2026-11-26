"""Rebuild docs/chapters/*.html & docs/index.html with SynthWave '84 neon vaporwave theme."""
from pathlib import Path

CHAPTERS = [
    ("examples/05_deep_learning/00_the_perceptron.ipynb",         "00_the_perceptron.html",            "Foundations &middot; The Perceptron"),
    ("examples/tldr_end2end/01_housing_classical_ml.ipynb",       "01_housing_classical_ml.html",      "Ch. 1 &middot; Housing Classical ML"),
    ("examples/tldr_end2end/02_housing_neural_network.ipynb",     "02_housing_neural_network.html",    "Ch. 2 &middot; Housing Neural Net"),
    ("examples/tldr_end2end/03_fashion_classical_ml.ipynb",       "03_fashion_classical_ml.html",      "Ch. 3 &middot; Fashion Classical CV"),
    ("examples/tldr_end2end/04_fashion_cnn_deep_learning.ipynb",  "04_fashion_cnn_deep_learning.html", "Ch. 4 &middot; Fashion 2D CNN"),
]

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
    --jp-mirror-editor-builtin-color: #ff7edb !important;
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
    background: #1e1a2b !important;
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
  .highlight .k, .highlight .kn, .highlight .kd, .highlight .kc, .highlight .ow {
    color: #ff7edb !important; font-weight: 600;
  }
  .highlight .o { color: #ff7edb !important; }
  .highlight .nn, .highlight .nc, .highlight .nb, .highlight .bp {
    color: #36f9f6 !important; font-weight: 500;
  }
  .highlight .nf, .highlight .fm, .highlight .nd {
    color: #72f1b8 !important; font-weight: 600;
  }
  .highlight .s, .highlight .s1, .highlight .s2, .highlight .sa, .highlight .sb, .highlight .sc, .highlight .sd, .highlight .se, .highlight .si {
    color: #ff9e64 !important;
  }
  .highlight .m, .highlight .mi, .highlight .mf, .highlight .mh, .highlight .mo {
    color: #fede5d !important;
  }
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
    padding: 5px 11px; border-radius: 6px; background: #2a2139;
    border: 1px solid rgba(54, 249, 246, 0.45);
    text-shadow: 0 0 6px rgba(54, 249, 246, 0.35);
  }
  .vkbook-nav a:hover {
    background: #382c4f; border-color: #ff7edb; color: #ff7edb;
  }
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

chap_dir = Path("docs/chapters")
for idx, (_, fname, title) in enumerate(CHAPTERS):
    fp = chap_dir / fname
    html = fp.read_text(encoding="utf-8")
    prev_link = f'<a href="{CHAPTERS[idx-1][1]}">&larr; Prev</a>' if idx > 0 else ""
    next_link = f'<a href="{CHAPTERS[idx+1][1]}">Next &rarr;</a>' if idx + 1 < len(CHAPTERS) else ""
    nav_html = (
        f'<div class="vkbook-nav">'
        f'<a href="../index.html">&larr; Book Home</a>'
        f'<span class="title">{title}</span>'
        f'<span style="display:flex;gap:6px;">{prev_link}{next_link}</span>'
        f'</div>'
    )
    footer_html = (
        f'<div class="vkbook-footer">'
        f'{prev_link or "<a href=\"../index.html\">&larr; Book Home</a>"}'
        f'{next_link or "<a href=\"../index.html\">Back to Table of Contents &rarr;</a>"}'
        f'</div>'
    )
    if "vkbook-nav" in html:
        continue
    html = html.replace("</head>", SYNTHWAVE84_CSS + "\n</head>", 1)
    bi = html.find(">", html.find("<body")) + 1
    html = html[:bi] + "\n" + nav_html + "\n" + html[bi:]
    html = html.replace("</body>", "\n" + footer_html + "\n</body>", 1)
    fp.write_text(html, encoding="utf-8")
    print("Styled SynthWave '84 ->", fp)

index_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hands-On AI, ML &amp; GPU Computing — Without CUDA or ROCm (vkdata)</title>
  <style>
    :root {
      --bg: #262335; --card: #1e1a2b; --border: #495495;
      --text: #f4eee4; --muted: #b6b1d6;
      --pink: #ff7edb; --cyan: #36f9f6; --yellow: #fede5d; --mint: #72f1b8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background-color: var(--bg); color: var(--text);
      background-image:
        radial-gradient(circle at 18% 0%, rgba(255, 126, 219, 0.16), transparent 42%),
        radial-gradient(circle at 85% 5%, rgba(54, 249, 246, 0.14), transparent 38%),
        linear-gradient(180deg, #241b2f 0%, #262335 45%, #1c1828 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.6; min-height: 100vh;
    }
    .wrap { max-width: 860px; margin: 0 auto; padding: 28px 18px 60px; }
    .badge {
      display: inline-block; font-size: 12px; font-weight: 700;
      color: var(--cyan); border: 1px solid rgba(54, 249, 246, 0.55);
      background: rgba(54, 249, 246, 0.1); padding: 4px 12px; border-radius: 999px;
      margin-bottom: 14px; text-shadow: 0 0 8px rgba(54, 249, 246, 0.45);
      box-shadow: 0 0 14px rgba(54, 249, 246, 0.15);
    }
    h1 {
      font-size: clamp(26px, 5vw, 40px); line-height: 1.18; margin: 0 0 12px;
      color: var(--pink);
      text-shadow: 0 0 18px rgba(255, 126, 219, 0.55);
    }
    .sub { color: var(--muted); font-size: 16px; margin: 0 0 26px; }
    h2 {
      font-size: 20px; margin: 32px 0 14px; color: var(--cyan);
      text-shadow: 0 0 10px rgba(54, 249, 246, 0.4);
      border-bottom: 1px solid rgba(255, 126, 219, 0.35); padding-bottom: 6px;
    }
    .grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
    .card {
      display: block; text-decoration: none; color: inherit;
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px 18px;
      transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
    }
    .card:hover {
      border-color: var(--pink); transform: translateY(-2px);
      box-shadow: 0 0 22px rgba(255, 126, 219, 0.25);
    }
    .card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
    .ch {
      color: var(--yellow); font-weight: 700; font-size: 13px;
      text-transform: uppercase; letter-spacing: 0.05em;
      text-shadow: 0 0 6px rgba(254, 222, 93, 0.35);
    }
    .pill {
      font-size: 12px; font-weight: 700; color: var(--mint);
      background: rgba(114, 241, 184, 0.12);
      border: 1px solid rgba(114, 241, 184, 0.35);
      padding: 2px 9px; border-radius: 6px;
    }
    .card h3 { margin: 6px 0 6px; font-size: 18px; color: #ffffff; }
    .card p { margin: 0; color: var(--muted); font-size: 14.5px; }
    .note {
      margin-top: 30px; background: #1a1626; border: 1px solid rgba(54, 249, 246, 0.4);
      border-radius: 12px; padding: 15px 18px; font-size: 14px; color: var(--muted);
      box-shadow: 0 0 18px rgba(54, 249, 246, 0.1);
    }
    .note strong { color: var(--yellow); }
    .note a { color: var(--cyan); text-decoration: none; font-weight: 600; }
    code {
      background: #2b2340; padding: 2px 6px; border-radius: 5px;
      font-size: 13px; color: var(--pink);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="badge">Interactive Follow-Along Book &middot; Pure Vulkan GPU</div>
    <h1>Hands-On AI, Machine Learning &amp; GPU Computing &mdash; Without CUDA or ROCm</h1>
    <p class="sub">
      Learn how AI actually works on the Vulkan GPU you already own (Steam Deck, AMD APU, Intel iGPU/Arc, older Radeon/NVIDIA, or Apple MoltenVK) &mdash; powered by the bundled <code>vkdata</code> toolkit.
    </p>

    <h2>Start Here &middot; Building Blocks of AI</h2>
    <div class="grid">
      <a class="card" href="chapters/00_the_perceptron.html">
        <div class="card-head">
          <span class="ch">Foundations &middot; 1 Artificial Neuron</span>
          <span class="pill">AND / OR &rarr; GPU Neuron &rarr; XOR Wall</span>
        </div>
        <h3>The Perceptron: How a Single Artificial Neuron Works</h3>
        <p>Start from zero: weighted votes, watching a single straight line learn with <code>vklearn.Perceptron</code>, running the same neuron on your Vulkan GPU with <code>vktorch</code>, and hitting the famous XOR limit.</p>
      </a>
    </div>

    <h2>Part I &middot; End-to-End Projects: Classical ML vs. Deep Learning</h2>
    <div class="grid">
      <a class="card" href="chapters/01_housing_classical_ml.html">
        <div class="card-head">
          <span class="ch">Chapter 1 &middot; Regression</span>
          <span class="pill">$69.6k &rarr; $52.7k RMSE</span>
        </div>
        <h3>California Housing: Feature Discovery &amp; Classical ML</h3>
        <p>Load 20,640 Census districts, spot the $500k cap on a GPU map, engineer 36 domain + Haversine + KMeans-RBF features, and race LinearRegression, Ridge, and GPU KNN.</p>
      </a>

      <a class="card" href="chapters/02_housing_neural_network.html">
        <div class="card-head">
          <span class="ch">Chapter 2 &middot; Deep Learning</span>
          <span class="pill">$48.2k RMSE &middot; R&sup2; 0.828</span>
        </div>
        <h3>California Housing: Beating Classical ML with a Deep Neural Network</h3>
        <p>Train a <code>vktorch</code> Pre-Norm Residual MLP (fused SPIR-V LayerNorm, GELU, AdamW) on the exact same split &mdash; beating GPU KNN by -$4,510 RMSE with .npz checkpointing.</p>
      </a>

      <a class="card" href="chapters/03_fashion_classical_ml.html">
        <div class="card-head">
          <span class="ch">Chapter 3 &middot; Classical CV</span>
          <span class="pill">81.5% &rarr; 86.45% Acc</span>
        </div>
        <h3>Fashion-MNIST: Image Feature Discovery &amp; Classical Computer Vision</h3>
        <p>Visualize GPU &ldquo;Average Clothing&rdquo; prototypes, discover Eigen-Clothes with PCA, hand-craft Sobel/HOG edge features, and analyze the 10&times;10 confusion matrix.</p>
      </a>

      <a class="card" href="chapters/04_fashion_cnn_deep_learning.html">
        <div class="card-head">
          <span class="ch">Chapter 4 &middot; 2D CNN</span>
          <span class="pill">87.85% Acc &middot; Shirt +11.5 pts</span>
        </div>
        <h3>Fashion-MNIST: Beating Classical CV with a 2D CNN (<code>vktorch</code>)</h3>
        <p>Train a strided <code>Conv2d</code> CNN directly on raw 28&times;28 pixels in 46s on Steam Deck GPU, inspect the 16 learned 3&times;3 filters &amp; feature maps, and beat every Classical CV model.</p>
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
