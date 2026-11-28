"""vkpandas: Vulkan GPU DataFrame library built on top of vknumpy."""

from . import accel
from .dataframe import DataFrame, GroupBy
from .io import concat, read_csv, read_parquet
from .series import Rolling, Series
from .streaming import StreamFrame, StreamGroupBy, recommended_vram_budget_mb, scan_csv

__all__ = [
    "DataFrame",
    "GroupBy",
    "Rolling",
    "Series",
    "StreamFrame",
    "StreamGroupBy",
    "accel",
    "concat",
    "read_csv",
    "read_parquet",
    "recommended_vram_budget_mb",
    "scan_csv",
]

__version__ = "0.1.0"
