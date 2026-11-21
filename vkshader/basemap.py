"""Esri World Imagery tile fetcher with VRAM mosaic caching for vkshader."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import requests
import vknumpy as vnp

_MERCATOR_HALF = 20037508.342789244
_ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_HTTP_SESSION = requests.Session()
_TILE_MEM_CACHE: Dict[Tuple[int, int, int], Image.Image] = {}
_MOSAIC_GPU_CACHE: Dict[Tuple[int, int, int, int, int], Tuple[vnp.ndarray, float, float, float, float]] = {}


def get_esri_mosaic_gpu(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    width: int,
    height: int,
    zoom: Optional[int] = None,
    cache_dir: Path = Path("/tmp/vkshader_esri_cache"),
) -> Optional[Tuple[vnp.ndarray, Tuple[float, float, float, float]]]:
    """Fetch/cache an Esri World Imagery mosaic in GPU VRAM and return (bg_gpu_i32, uv_rect)."""
    alt_caches = (Path("/tmp/vkpandas_esri_cache"), Path("/tmp/vupy_esri_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if zoom is None:
        span_x = max(100.0, x_range[1] - x_range[0])
        zoom = int(max(10, min(17, round(math.log2(2.0 * _MERCATOR_HALF / span_x * (width / 256.0))))))

    n = 2.0 ** zoom
    tx0 = int(math.floor((x_range[0] + _MERCATOR_HALF) / (2.0 * _MERCATOR_HALF) * n))
    tx1 = int(math.floor((x_range[1] + _MERCATOR_HALF) / (2.0 * _MERCATOR_HALF) * n))
    ty0 = int(math.floor((_MERCATOR_HALF - y_range[1]) / (2.0 * _MERCATOR_HALF) * n))
    ty1 = int(math.floor((_MERCATOR_HALF - y_range[0]) / (2.0 * _MERCATOR_HALF) * n))

    mosaic_key = (zoom, tx0, tx1, ty0, ty1)
    cached = _MOSAIC_GPU_CACHE.get(mosaic_key)
    if cached is not None:
        bg_gpu, bx0, bx1, by0, by1 = cached
    else:
        canvas = Image.new("RGBA", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_key = (zoom, ty, tx)
                tile = _TILE_MEM_CACHE.get(tile_key)
                if tile is None:
                    fname = f"esri_{zoom}_{ty}_{tx}.jpg"
                    found_path = None
                    for d in (cache_dir, *alt_caches):
                        cand = d / fname
                        if cand.is_file():
                            found_path = cand
                            break
                    if found_path is not None:
                        tile = Image.open(found_path).convert("RGBA")
                    else:
                        try:
                            resp = _HTTP_SESSION.get(
                                _ESRI_URL.format(z=zoom, y=ty, x=tx),
                                timeout=5,
                                headers={"User-Agent": "vkshader/0.1"},
                            )
                            resp.raise_for_status()
                        except requests.RequestException:
                            return None
                        (cache_dir / fname).write_bytes(resp.content)
                        tile = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    if len(_TILE_MEM_CACHE) > 256:
                        _TILE_MEM_CACHE.clear()
                    _TILE_MEM_CACHE[tile_key] = tile
                canvas.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))

        bx0 = tx0 / n * (2.0 * _MERCATOR_HALF) - _MERCATOR_HALF
        bx1 = (tx1 + 1) / n * (2.0 * _MERCATOR_HALF) - _MERCATOR_HALF
        by1 = _MERCATOR_HALF - ty0 / n * (2.0 * _MERCATOR_HALF)
        by0 = _MERCATOR_HALF - (ty1 + 1) / n * (2.0 * _MERCATOR_HALF)

        rgba_np = np.ascontiguousarray(canvas, dtype=np.uint8)
        packed_i32 = rgba_np.view(np.int32).reshape(rgba_np.shape[0], rgba_np.shape[1])
        bg_gpu = vnp.array(packed_i32, dtype=vnp.int32)
        if len(_MOSAIC_GPU_CACHE) > 8:
            _MOSAIC_GPU_CACHE.clear()
        _MOSAIC_GPU_CACHE[mosaic_key] = (bg_gpu, bx0, bx1, by0, by1)

    uv_left = (x_range[0] - bx0) / (bx1 - bx0)
    uv_right = (x_range[1] - bx0) / (bx1 - bx0)
    uv_top = (by1 - y_range[1]) / (by1 - by0)
    uv_bottom = (by1 - y_range[0]) / (by1 - by0)
    return bg_gpu, (uv_left, uv_top, uv_right, uv_bottom)
