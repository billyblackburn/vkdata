"""Built-in text / vector dataset loaders for vkvec."""

from __future__ import annotations

import urllib.request
from pathlib import Path

BBC_NEWS_CLASSES = ["Business", "Entertainment", "Politics", "Sport", "Tech"]
_BBC_URL = (
    "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
)


def load_bbc_news(data_dir: str | Path | None = None):
    """Download/cache the 2,225 full-length BBC News articles as ``(df, class_names)``."""
    import vkpandas as vpd

    if data_dir is None:
        repo_cache = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "tldr_end2end"
            / "data"
            / "bbc_news"
        )
        cache_dir = repo_cache if repo_cache.parent.exists() else Path("data/bbc_news")
    else:
        cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "bbc-text.csv"
    if not csv_path.exists():
        fallback = cache_dir.parent / "bbc-text.csv"
        if fallback.exists():
            csv_path.write_bytes(fallback.read_bytes())
        else:
            urllib.request.urlretrieve(_BBC_URL, csv_path)
    return vpd.read_csv(csv_path), list(BBC_NEWS_CLASSES)


__all__ = ["BBC_NEWS_CLASSES", "load_bbc_news"]
