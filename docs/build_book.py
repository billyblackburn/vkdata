"""Rebuild docs/chapters/*.html from notebooks with dark theme, Python syntax colors, and mobile nav."""
from pathlib import Path
import subprocess

CHAPTERS = [
    ("examples/05_deep_learning/00_the_perceptron.ipynb",         "00_the_perceptron.html",          "Foundations &middot; The Perceptron"),
    ("examples/tldr_end2end/01_housing_classical_ml.ipynb",       "01_housing_classical_ml.html",    "Ch. 1 &middot; Housing Classical ML"),
    ("examples/tldr_end2end/02_housing_neural_network.ipynb",     "02_housing_neural_network.html",  "Ch. 2 &middot; Housing Neural Net"),
    ("examples/tldr_end2end/03_fashion_classical_ml.ipynb",       "03_fashion_classical_ml.html",    "Ch. 3 &middot; Fashion Classical CV"),
    ("examples/tldr_end2end/04_fashion_cnn_deep_learning.ipynb",  "04_fashion_cnn_deep_learning.html", "Ch. 4 &middot; Fashion 2D CNN"),
]

DARK_PYTHON_CSS = """
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --jp-layout-color0: #0d1117 !important;
    --jp-layout-color1: #161b22 !important;
    --jp-layout-color2: #1f2633 !important;
    --jp-border-color1: #30363d !important;
    --jp-border-color2: #262c36 !important;
    --jp-ui-font-color1: #e6edf3 !important;
    --jp-content-font-color1: #e6edf3 !important;
    --jp-cell-editor-background: #161b22 !important;
    --jp-cell-editor-border-color: #30363d !important;
  }
  html, body, body.jp-Notebook {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
  }
  .jp-RenderedMarkdown {
    color: #e6edf3 !important;
    line-height: 1.65;
  }
  .jp-RenderedMarkdown h1, .jp-RenderedMarkdown h2, .jp-RenderedMarkdown h3 {
    color: #f0f6fc !important;
    border-color: #30363d !important;
  }
  .jp-RenderedMarkdown a { color: #58a6ff !important; }
  .jp-RenderedMarkdown code {
    background: #1f2633 !important; color: #79c0ff !important;
    padding: 2px 6px; border-radius: 5px;
  }
  .jp-RenderedMarkdown blockquote {
    border-left: 3px solid #58a6ff !important;
    background: #131922; padding: 8px 14px; border-radius: 0 8px 8px 0;
    color: #c9d1d9 !important;
  }
  .jp-InputArea-editor {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
  }
  .jp-OutputArea-output pre {
    color: #c9d1d9 !important;
    background: #0f141b !important;
    border-radius: 6px; padding: 8px 10px;
  }

  /* Vibrant VS Code / GitHub Dark Python syntax highlighting (Pygments tokens) */
  .highlight { background: #161b22 !important; color: #e6edf3 !important; }
  .highlight .c, .highlight .c1, .highlight .cm, .highlight .ch { color: #8b949e !important; font-style: italic; }
  .highlight .k, .highlight .kn, .highlight .kd, .highlight .kc { color: #ff7b72 !important; font-weight: 600; }
  .highlight .ow { color: #ff7b72 !important; }
  .highlight .o { color: #d2a8ff !important; }
  .highlight .nf, .highlight .fm { color: #d2a8ff !important; font-weight: 600; }
  .highlight .nc { color: #ffa657 !important; font-weight: 600; }
  .highlight .nb, .highlight .bp { color: #79c0ff !important; }
  .highlight .s, .highlight .s1, .highlight .s2, .highlight .sa, .highlight .sb, .highlight .sc, .highlight .sd, .highlight .se, .highlight .si { color: #a5d6ff !important; }
  .highlight .m, .highlight .mi, .highlight .mf, .highlight .mh, .highlight .mo { color: #79c0ff !important; }
  .highlight .n { color: #e6edf3 !important; }
  .highlight .nd { color: #d2a8ff !important; }
  .highlight .nn { color: #ffa657 !important; }
  .highlight .p { color: #c9d1d9 !important; }

  /* Sticky dark top & bottom chapter navigation */
  .vkbook-nav {
    position: sticky; top: 0; z-index: 1000;
    background: #11151d; color: #e6edf3;
    border-bottom: 1px solid #30363d;
    padding: 10px 14px; display: flex; flex-wrap: wrap;
    align-items: center; justify-content: space-between; gap: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
  }
  .vkbook-nav a {
    color: #58a6ff; text-decoration: none; font-weight: 600;
    padding: 5px 11px; border-radius: 6px; background: #1c2331;
    border: 1px solid #30363d;
  }
  .vkbook-nav a:hover { background: #263145; }
  .vkbook-nav .title { color: #f0f6fc; font-weight: 600; }
  .vkbook-footer {
    margin: 32px 0 24px; padding: 14px; border-top: 1px solid #30363d;
    display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .vkbook-footer a {
    color: #58a6ff; text-decoration: none; font-weight: 600;
    padding: 8px 14px; background: #161b22;
    border: 1px solid #30363d; border-radius: 8px;
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
    html = html.replace("</head>", DARK_PYTHON_CSS + "\n</head>", 1)
    bi = html.find(">", html.find("<body")) + 1
    html = html[:bi] + "\n" + nav_html + "\n" + html[bi:]
    html = html.replace("</body>", "\n" + footer_html + "\n</body>", 1)
    fp.write_text(html, encoding="utf-8")
    print("Styled dark Python theme ->", fp)
