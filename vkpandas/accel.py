"""vkpandas.accel: Zero-code-change Vulkan GPU accelerator for pandas (like cudf.pandas).

Usage::

    import vkpandas.accel
    vkpandas.accel.install()

    import pandas as pd          # Transparently accelerated on Vulkan GPU!
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

In Jupyter notebooks::

    %load_ext vkpandas.accel
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
import time
import types
from contextlib import contextmanager
from typing import Any, Iterator

import vkpandas as _vpd


class _AccelStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.gpu_calls: int = 0
        self.cpu_fallbacks: int = 0
        self.ops: dict[str, dict[str, int]] = {}

    def record(self, op: str, *, gpu: bool) -> None:
        with self._lock:
            if gpu:
                self.gpu_calls += 1
            else:
                self.cpu_fallbacks += 1
            entry = self.ops.setdefault(op, {"gpu": 0, "cpu": 0})
            entry["gpu" if gpu else "cpu"] += 1

    def reset(self) -> None:
        with self._lock:
            self.gpu_calls = 0
            self.cpu_fallbacks = 0
            self.ops.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "gpu_calls": self.gpu_calls,
                "cpu_fallbacks": self.cpu_fallbacks,
                "ops": {k: dict(v) for k, v in self.ops.items()},
            }


_stats = _AccelStats()
_installed: bool = False
_orig_pandas: types.ModuleType | None = None
_real_pandas: types.ModuleType | None = None
_proxy_module: types.ModuleType | None = None
_meta_finder: "_PandasAccelFinder | None" = None


def _to_real_pandas(obj: Any) -> Any:
    if _real_pandas is None:
        return obj
    if isinstance(obj, _vpd.DataFrame):
        return _real_pandas.DataFrame({c: obj[c].to_list() for c in obj.columns})
    if isinstance(obj, _vpd.Series):
        return _real_pandas.Series(obj.to_list(), name=obj.name)
    return obj


def _from_real_pandas(obj: Any) -> Any:
    if _real_pandas is None:
        return obj
    if isinstance(obj, _real_pandas.DataFrame):
        return _vpd.DataFrame({c: obj[c].to_list() for c in obj.columns})
    if isinstance(obj, _real_pandas.Series):
        return _vpd.Series(obj.to_list(), name=obj.name)
    return obj


def _wrap_callable(name: str, gpu_fn: Any, cpu_fn: Any = None):
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            res = gpu_fn(*args, **kwargs)
            _stats.record(name, gpu=True)
            return res
        except (AttributeError, TypeError, NotImplementedError, ValueError):
            if cpu_fn is None:
                raise
            c_args = tuple(_to_real_pandas(a) for a in args)
            c_kwargs = {k: _to_real_pandas(v) for k, v in kwargs.items()}
            res = cpu_fn(*c_args, **c_kwargs)
            _stats.record(name, gpu=False)
            return _from_real_pandas(res)

    _wrapped.__name__ = name
    _wrapped.__doc__ = getattr(gpu_fn, "__doc__", None)
    return _wrapped


def _build_proxy_module(real_pd: types.ModuleType | None) -> types.ModuleType:
    mod = types.ModuleType("pandas")
    mod.__file__ = __file__
    mod.__package__ = "pandas"
    mod.__path__ = []
    mod.__vkdata_accel__ = True
    mod.__version__ = getattr(real_pd, "__version__", f"{_vpd.__version__}+vkpandas")

    gpu_exports = {
        "DataFrame": _vpd.DataFrame,
        "Series": _vpd.Series,
        "GroupBy": _vpd.GroupBy,
        "Rolling": _vpd.Rolling,
        "read_csv": _vpd.read_csv,
        "read_parquet": _vpd.read_parquet,
        "concat": _vpd.concat,
    }
    for name, gpu_obj in gpu_exports.items():
        cpu_obj = getattr(real_pd, name, None) if real_pd is not None else None
        if isinstance(gpu_obj, type):
            setattr(mod, name, _wrap_class(name, gpu_obj, cpu_obj))
        else:
            setattr(mod, name, _wrap_callable(f"pandas.{name}", gpu_obj, cpu_obj))

    def _module_getattr(attr: str) -> Any:
        if real_pd is not None and hasattr(real_pd, attr):
            val = getattr(real_pd, attr)
            if callable(val):
                def _cpu_only(*args: Any, **kwargs: Any) -> Any:
                    _stats.record(f"pandas.{attr}", gpu=False)
                    c_args = tuple(_to_real_pandas(a) for a in args)
                    c_kwargs = {k: _to_real_pandas(v) for k, v in kwargs.items()}
                    return _from_real_pandas(val(*c_args, **c_kwargs))
                return _cpu_only
            return val
        raise AttributeError(f"module 'pandas' (vkpandas.accel) has no attribute {attr!r}")

    mod.__getattr__ = _module_getattr  # type: ignore[method-assign]
    return mod


def _wrap_class(name: str, gpu_cls: type, cpu_cls: type | None) -> type:
    class _AccelProxy(gpu_cls):  # type: ignore[valid-type,misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            try:
                super().__init__(*args, **kwargs)
                _stats.record(f"pandas.{name}", gpu=True)
            except (TypeError, ValueError, NotImplementedError):
                if cpu_cls is None:
                    raise
                raw = cpu_cls(*args, **kwargs)
                _stats.record(f"pandas.{name}", gpu=False)
                converted = _from_real_pandas(raw)
                self.__dict__.update(converted.__dict__)

        def __getattr__(self, attr: str) -> Any:
            if cpu_cls is None or _real_pandas is None:
                raise AttributeError(f"{name!r} object has no attribute {attr!r}")
            cpu_obj = _to_real_pandas(self)
            if not hasattr(cpu_obj, attr):
                raise AttributeError(f"{name!r} object has no attribute {attr!r}")
            target = getattr(cpu_obj, attr)
            if callable(target):
                def _fallback(*args: Any, **kwargs: Any) -> Any:
                    _stats.record(f"{name}.{attr}", gpu=False)
                    c_args = tuple(_to_real_pandas(a) for a in args)
                    c_kwargs = {k: _to_real_pandas(v) for k, v in kwargs.items()}
                    return _from_real_pandas(target(*c_args, **c_kwargs))
                return _fallback
            _stats.record(f"{name}.{attr}", gpu=False)
            return _from_real_pandas(target)

    _AccelProxy.__name__ = name
    _AccelProxy.__qualname__ = name
    _AccelProxy.__doc__ = gpu_cls.__doc__
    return _AccelProxy


class _PandasAccelLoader(importlib.abc.Loader):
    def create_module(self, spec: Any) -> types.ModuleType:
        assert _proxy_module is not None
        return _proxy_module

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _PandasAccelFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == "pandas":
            return importlib.machinery.ModuleSpec("pandas", _PandasAccelLoader(), is_package=True)
        return None


def install() -> None:
    """Activate zero-code-change Vulkan GPU acceleration for ``import pandas``."""
    global _installed, _orig_pandas, _real_pandas, _proxy_module, _meta_finder
    if _installed:
        return

    _orig_pandas = sys.modules.get("pandas")
    if _orig_pandas is not None and not getattr(_orig_pandas, "__vkdata_accel__", False):
        _real_pandas = _orig_pandas
    else:
        try:
            _real_pandas = importlib.import_module("pandas")
        except ImportError:
            _real_pandas = None

    _proxy_module = _build_proxy_module(_real_pandas)
    sys.modules["pandas"] = _proxy_module
    _meta_finder = _PandasAccelFinder()
    sys.meta_path.insert(0, _meta_finder)
    _installed = True


def uninstall() -> None:
    """Deactivate vkpandas.accel and restore the original pandas module state."""
    global _installed, _meta_finder, _proxy_module
    if not _installed:
        return
    if _meta_finder in sys.meta_path:
        sys.meta_path.remove(_meta_finder)
    _meta_finder = None
    if _orig_pandas is not None:
        sys.modules["pandas"] = _orig_pandas
    else:
        sys.modules.pop("pandas", None)
    _proxy_module = None
    _installed = False


def is_installed() -> bool:
    """Return True if vkpandas.accel is currently intercepting ``import pandas``."""
    return _installed


def stats() -> dict[str, Any]:
    """Return GPU vs. CPU fallback call counts recorded by vkpandas.accel."""
    return _stats.snapshot()


def reset_stats() -> None:
    """Reset vkpandas.accel GPU/CPU call counters."""
    _stats.reset()


@contextmanager
def profile(*, verbose: bool = True) -> Iterator[dict[str, Any]]:
    """Context manager summarizing Vulkan GPU vs. CPU fallback calls (like cudf.pandas.profile)."""
    before = _stats.snapshot()
    t0 = time.perf_counter()
    report: dict[str, Any] = {}
    try:
        yield report
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        after = _stats.snapshot()
        gpu_n = after["gpu_calls"] - before["gpu_calls"]
        cpu_n = after["cpu_fallbacks"] - before["cpu_fallbacks"]
        report.update(
            {
                "gpu_calls": gpu_n,
                "cpu_fallbacks": cpu_n,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )
        if verbose:
            total = max(1, gpu_n + cpu_n)
            print(
                f"[vkpandas.accel] GPU calls: {gpu_n} ({gpu_n / total:.0%}) | "
                f"CPU fallbacks: {cpu_n} | {elapsed_ms:.2f} ms"
            )


def load_ipython_extension(ipython: Any = None) -> None:
    """Enable ``%load_ext vkpandas.accel`` in Jupyter notebooks."""
    install()


def unload_ipython_extension(ipython: Any = None) -> None:
    """Enable ``%unload_ext vkpandas.accel`` in Jupyter notebooks."""
    uninstall()


__all__ = [
    "install",
    "is_installed",
    "load_ipython_extension",
    "profile",
    "reset_stats",
    "stats",
    "uninstall",
    "unload_ipython_extension",
]
