"""vklearn.accel: Zero-code-change Vulkan GPU accelerator for scikit-learn (like cuml.accel).

Usage::

    import vklearn.accel
    vklearn.accel.install()

    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    # All transparently accelerated on your Vulkan GPU!

In Jupyter notebooks::

    %load_ext vklearn.accel
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

import vklearn as _vkl


class _SklearnAccelStats:
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


_stats = _SklearnAccelStats()
_installed: bool = False
_orig_modules: dict[str, types.ModuleType | None] = {}
_proxy_modules: dict[str, types.ModuleType] = {}
_meta_finder: "_SklearnAccelFinder | None" = None

def _get_submodule_map() -> dict[str, Any]:
    return {
        "sklearn.linear_model": _vkl.linear_model,
        "sklearn.neighbors": _vkl.neighbors,
        "sklearn.decomposition": _vkl.decomposition,
        "sklearn.cluster": _vkl.cluster,
        "sklearn.preprocessing": _vkl.preprocessing,
        "sklearn.metrics": _vkl.metrics,
        "sklearn.model_selection": _vkl.model_selection,
        "sklearn.pipeline": _vkl.pipeline,
        "sklearn.datasets": _vkl.datasets,
    }


def _wrap_estimator(full_name: str, gpu_cls: type, cpu_cls: type | None) -> type:
    class _GpuEstimatorProxy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            try:
                self._impl = gpu_cls(*args, **kwargs)
                self._on_gpu = True
                _stats.record(full_name, gpu=True)
            except (TypeError, ValueError, NotImplementedError):
                if cpu_cls is None:
                    raise
                self._impl = cpu_cls(*args, **kwargs)
                self._on_gpu = False
                _stats.record(full_name, gpu=False)

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._impl, attr)

        def __repr__(self) -> str:
            tag = "VulkanGPU" if self._on_gpu else "CPUFallback"
            return f"<{full_name} [{tag}]: {self._impl!r}>"

    _GpuEstimatorProxy.__name__ = gpu_cls.__name__
    _GpuEstimatorProxy.__qualname__ = gpu_cls.__name__
    _GpuEstimatorProxy.__doc__ = gpu_cls.__doc__
    return _GpuEstimatorProxy


def _wrap_func(full_name: str, gpu_fn: Any, cpu_fn: Any = None) -> Any:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            res = gpu_fn(*args, **kwargs)
            _stats.record(full_name, gpu=True)
            return res
        except (TypeError, ValueError, NotImplementedError):
            if cpu_fn is None:
                raise
            _stats.record(full_name, gpu=False)
            return cpu_fn(*args, **kwargs)

    _wrapped.__name__ = getattr(gpu_fn, "__name__", full_name)
    _wrapped.__doc__ = getattr(gpu_fn, "__doc__", None)
    return _wrapped


def _build_submodule(mod_name: str, gpu_mod: types.ModuleType, real_mod: types.ModuleType | None) -> types.ModuleType:
    proxy = types.ModuleType(mod_name)
    proxy.__file__ = __file__
    proxy.__package__ = "sklearn"
    proxy.__vkdata_accel__ = True

    export_names = getattr(gpu_mod, "__all__", [n for n in dir(gpu_mod) if not n.startswith("_")])
    for name in export_names:
        if not hasattr(gpu_mod, name):
            continue
        gpu_obj = getattr(gpu_mod, name)
        cpu_obj = getattr(real_mod, name, None) if real_mod is not None else None
        full_name = f"{mod_name}.{name}"
        if isinstance(gpu_obj, type):
            setattr(proxy, name, _wrap_estimator(full_name, gpu_obj, cpu_obj))
        elif callable(gpu_obj):
            setattr(proxy, name, _wrap_func(full_name, gpu_obj, cpu_obj))
        else:
            setattr(proxy, name, gpu_obj)

    def _sub_getattr(attr: str) -> Any:
        if real_mod is not None and hasattr(real_mod, attr):
            val = getattr(real_mod, attr)
            if callable(val):
                def _cpu_only(*args: Any, **kwargs: Any) -> Any:
                    _stats.record(f"{mod_name}.{attr}", gpu=False)
                    return val(*args, **kwargs)
                return _cpu_only
            return val
        raise AttributeError(f"module {mod_name!r} (vklearn.accel) has no attribute {attr!r}")

    proxy.__getattr__ = _sub_getattr  # type: ignore[method-assign]
    return proxy


class _SklearnAccelLoader(importlib.abc.Loader):
    def __init__(self, fullname: str) -> None:
        self._fullname = fullname

    def create_module(self, spec: Any) -> types.ModuleType:
        return _proxy_modules[self._fullname]

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _SklearnAccelFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname in _proxy_modules:
            is_pkg = fullname == "sklearn"
            return importlib.machinery.ModuleSpec(fullname, _SklearnAccelLoader(fullname), is_package=is_pkg)
        return None


def install() -> None:
    """Activate zero-code-change Vulkan GPU acceleration for ``import sklearn``."""
    global _installed, _meta_finder
    if _installed:
        return

    _orig_modules.clear()
    _proxy_modules.clear()

    # Probe real sklearn if installed (without failing if absent)
    real_root: types.ModuleType | None = None
    _orig_modules["sklearn"] = sys.modules.get("sklearn")
    if _orig_modules["sklearn"] is not None and not getattr(_orig_modules["sklearn"], "__vkdata_accel__", False):
        real_root = _orig_modules["sklearn"]
    else:
        try:
            real_root = importlib.import_module("sklearn")
        except ImportError:
            real_root = None

    root_proxy = types.ModuleType("sklearn")
    root_proxy.__file__ = __file__
    root_proxy.__package__ = "sklearn"
    root_proxy.__path__ = []
    root_proxy.__vkdata_accel__ = True
    root_proxy.__version__ = getattr(real_root, "__version__", "0.1.0+vklearn")
    _proxy_modules["sklearn"] = root_proxy

    for sub_name, gpu_mod in _get_submodule_map().items():
        _orig_modules[sub_name] = sys.modules.get(sub_name)
        real_sub: types.ModuleType | None = None
        if real_root is not None:
            try:
                real_sub = importlib.import_module(sub_name)
            except ImportError:
                real_sub = None
        sub_proxy = _build_submodule(sub_name, gpu_mod, real_sub)
        _proxy_modules[sub_name] = sub_proxy
        setattr(root_proxy, sub_name.split(".", 1)[1], sub_proxy)

    for name, mod in _proxy_modules.items():
        sys.modules[name] = mod

    _meta_finder = _SklearnAccelFinder()
    sys.meta_path.insert(0, _meta_finder)
    _installed = True


def uninstall() -> None:
    """Deactivate vklearn.accel and restore the original sklearn module state."""
    global _installed, _meta_finder
    if not _installed:
        return
    if _meta_finder in sys.meta_path:
        sys.meta_path.remove(_meta_finder)
    _meta_finder = None
    for name, orig in _orig_modules.items():
        if orig is not None:
            sys.modules[name] = orig
        else:
            sys.modules.pop(name, None)
    _orig_modules.clear()
    _proxy_modules.clear()
    _installed = False


def is_installed() -> bool:
    """Return True if vklearn.accel is currently intercepting ``import sklearn``."""
    return _installed


def stats() -> dict[str, Any]:
    """Return GPU vs. CPU fallback call counts recorded by vklearn.accel."""
    return _stats.snapshot()


def reset_stats() -> None:
    """Reset vklearn.accel GPU/CPU call counters."""
    _stats.reset()


@contextmanager
def profile(*, verbose: bool = True) -> Iterator[dict[str, Any]]:
    """Context manager summarizing Vulkan GPU vs. CPU fallback calls (like cuml.accel)."""
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
                f"[vklearn.accel] GPU calls: {gpu_n} ({gpu_n / total:.0%}) | "
                f"CPU fallbacks: {cpu_n} | {elapsed_ms:.2f} ms"
            )


def load_ipython_extension(ipython: Any = None) -> None:
    """Enable ``%load_ext vklearn.accel`` in Jupyter notebooks."""
    install()


def unload_ipython_extension(ipython: Any = None) -> None:
    """Enable ``%unload_ext vklearn.accel`` in Jupyter notebooks."""
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
