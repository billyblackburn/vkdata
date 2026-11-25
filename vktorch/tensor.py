"""vktorch.tensor: Vulkan GPU Tensor with dynamic reverse-mode automatic differentiation."""

from __future__ import annotations

import contextlib
import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as _np
import vknumpy as vnp

_GRAD_ENABLED: bool = True


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


@contextlib.contextmanager
def no_grad():
    """Context manager that disables gradient tracking (like ``torch.no_grad``)."""
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def _to_vnp(data, dtype=_np.float32) -> vnp.ndarray:
    if isinstance(data, Tensor):
        arr = data.data
        return arr if (dtype is None or arr.dtype == dtype) else arr.astype(dtype)
    if isinstance(data, vnp.ndarray):
        return data if (dtype is None or data.dtype == dtype) else data.astype(dtype)
    np_arr = _np.asarray(data, dtype=dtype if dtype is not None else _np.float32)
    return vnp.array(np_arr, dtype=np_arr.dtype)


def _unbroadcast(grad: vnp.ndarray, target_shape: Tuple[int, ...]) -> vnp.ndarray:
    """Reduce ``grad`` along broadcasted dimensions to match ``target_shape``."""
    if grad.shape == target_shape:
        return grad
    if len(target_shape) == 0:
        return vnp.sum(grad).reshape(())

    g = grad
    # Sum out extra leading dimensions
    ndiff = g.ndim - len(target_shape)
    if ndiff > 0:
        g = vnp.sum(g, axis=tuple(range(ndiff)))

    # Sum out axes where target_shape has size 1
    keep_axes = tuple(i for i, (gs, ts) in enumerate(zip(g.shape, target_shape)) if ts == 1 and gs != 1)
    if keep_axes:
        g = vnp.sum(g, axis=keep_axes, keepdims=True)
    return g.reshape(target_shape) if g.shape != target_shape else g


def _swap_last2(a: vnp.ndarray) -> vnp.ndarray:
    if a.ndim < 2:
        return a
    perm = list(range(a.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return a.transpose(perm).contiguous()


class Tensor:
    """PyTorch-style Vulkan GPU Tensor backed by ``vknumpy.ndarray`` with reverse-mode autograd."""

    def __init__(
        self,
        data,
        *,
        requires_grad: bool = False,
        dtype=_np.float32,
        _parents: Tuple["Tensor", ...] = (),
        _backward: Optional[Callable[[vnp.ndarray], None]] = None,
    ) -> None:
        self.data: vnp.ndarray = _to_vnp(data, dtype=dtype)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[vnp.ndarray] = None
        self._parents: Tuple[Tensor, ...] = _parents
        self._backward_fn: Optional[Callable[[vnp.ndarray], None]] = _backward

    # --- Properties & conversions ---
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> _np.ndarray:
        return self.data.to_numpy()

    def to_numpy(self) -> _np.ndarray:
        return self.data.to_numpy()

    def item(self) -> float:
        return float(self.data.to_numpy().ravel()[0])

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False, dtype=None)

    def zero_grad(self) -> None:
        self.grad = None

    def _accum_grad(self, g: vnp.ndarray) -> None:
        if not self.requires_grad:
            return
        g = _unbroadcast(g, self.shape)
        self.grad = g if self.grad is None else (self.grad + g)

    def __repr__(self) -> str:
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}{rg})"

    # --- Reverse-mode autograd engine ---
    def backward(self, grad: Optional[Union["Tensor", vnp.ndarray, _np.ndarray]] = None) -> None:
        """Compute gradients of this tensor w.r.t. all ancestor leaf tensors with ``requires_grad=True``."""
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a Tensor with requires_grad=False")

        topo: List[Tensor] = []
        visited = set()

        def _build(node: Tensor) -> None:
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            for p in node._parents:
                _build(p)
            topo.append(node)

        _build(self)

        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be specified for non-scalar Tensor.backward()")
            g0 = vnp.ones(self.shape, dtype=_np.float32)
        else:
            g0 = _to_vnp(grad, dtype=_np.float32)

        self.grad = g0
        for node in reversed(topo):
            if node.grad is not None and node._backward_fn is not None:
                node._backward_fn(node.grad)

    # --- Arithmetic ops ---
    def _wrap(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False, dtype=_np.float32)

    def __add__(self, other) -> "Tensor":
        other_t = self._wrap(other)
        out_data = self.data + other_t.data
        req = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)

        def _bw(g: vnp.ndarray) -> None:
            if self.requires_grad:
                self._accum_grad(g)
            if other_t.requires_grad:
                other_t._accum_grad(g)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self, other_t) if req else (), _backward=_bw if req else None)

    def __radd__(self, other) -> "Tensor":
        return self._wrap(other) + self

    def __sub__(self, other) -> "Tensor":
        other_t = self._wrap(other)
        out_data = self.data - other_t.data
        req = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)

        def _bw(g: vnp.ndarray) -> None:
            if self.requires_grad:
                self._accum_grad(g)
            if other_t.requires_grad:
                other_t._accum_grad(-g)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self, other_t) if req else (), _backward=_bw if req else None)

    def __rsub__(self, other) -> "Tensor":
        return self._wrap(other) - self

    def __neg__(self) -> "Tensor":
        out_data = -self.data
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(-g)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def __mul__(self, other) -> "Tensor":
        other_t = self._wrap(other)
        out_data = self.data * other_t.data
        req = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)

        def _bw(g: vnp.ndarray) -> None:
            if self.requires_grad:
                self._accum_grad(g * other_t.data)
            if other_t.requires_grad:
                other_t._accum_grad(g * self.data)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self, other_t) if req else (), _backward=_bw if req else None)

    def __rmul__(self, other) -> "Tensor":
        return self._wrap(other) * self

    def __truediv__(self, other) -> "Tensor":
        other_t = self._wrap(other)
        out_data = self.data / other_t.data
        req = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)

        def _bw(g: vnp.ndarray) -> None:
            if self.requires_grad:
                self._accum_grad(g / other_t.data)
            if other_t.requires_grad:
                other_t._accum_grad(-(g * self.data) / (other_t.data * other_t.data))

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self, other_t) if req else (), _backward=_bw if req else None)

    def __rtruediv__(self, other) -> "Tensor":
        return self._wrap(other) / self

    def __pow__(self, power: float) -> "Tensor":
        p = float(power)
        if p == 2.0:
            out_data = self.data * self.data
        else:
            out_data = vnp.exp(vnp.log(vnp.maximum(self.data, 1e-12)) * p)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            if p == 2.0:
                self._accum_grad(g * (self.data * 2.0))
            else:
                self._accum_grad(g * (p * vnp.exp(vnp.log(vnp.maximum(self.data, 1e-12)) * (p - 1.0))))

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def __matmul__(self, other) -> "Tensor":
        other_t = self._wrap(other)
        out_data = self.data @ other_t.data
        req = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)

        def _bw(g: vnp.ndarray) -> None:
            if self.requires_grad:
                self._accum_grad(g @ _swap_last2(other_t.data))
            if other_t.requires_grad:
                other_t._accum_grad(_swap_last2(self.data) @ g)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self, other_t) if req else (), _backward=_bw if req else None)

    # --- Shape & movement ops ---
    def reshape(self, *shape) -> "Tensor":
        new_shape = shape[0] if (len(shape) == 1 and isinstance(shape[0], (tuple, list))) else shape
        out_data = self.data.contiguous().reshape(new_shape)
        req = _GRAD_ENABLED and self.requires_grad
        orig_shape = self.shape

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(g.contiguous().reshape(orig_shape))

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    view = reshape

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 0:
            perm = tuple(reversed(range(self.ndim)))
        elif len(axes) == 2 and self.ndim > 2:
            # PyTorch-style swap of two dims: x.transpose(1, 2)
            a0 = axes[0] + self.ndim if axes[0] < 0 else axes[0]
            a1 = axes[1] + self.ndim if axes[1] < 0 else axes[1]
            perm_list = list(range(self.ndim))
            perm_list[a0], perm_list[a1] = perm_list[a1], perm_list[a0]
            perm = tuple(perm_list)
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = tuple(axes)

        out_data = self.data.transpose(perm).contiguous()
        req = _GRAD_ENABLED and self.requires_grad
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        inv_perm = tuple(inv)

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(g.transpose(inv_perm).contiguous())

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def permute(self, *axes) -> "Tensor":
        perm = axes[0] if (len(axes) == 1 and isinstance(axes[0], (tuple, list))) else axes
        return self.transpose(tuple(perm))

    # --- Reductions ---
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out_data = vnp.sum(self.data, axis=axis, keepdims=keepdims)
        req = _GRAD_ENABLED and self.requires_grad
        orig_shape = self.shape

        def _bw(g: vnp.ndarray) -> None:
            if axis is None:
                self._accum_grad(vnp.ones(orig_shape, dtype=_np.float32) * g)
            else:
                g_exp = g
                if not keepdims:
                    ax_tuple = (axis,) if isinstance(axis, int) else tuple(axis)
                    ax_norm = sorted((a + len(orig_shape) if a < 0 else a) for a in ax_tuple)
                    sh = list(g.shape)
                    for a in ax_norm:
                        sh.insert(a, 1)
                    g_exp = g.reshape(tuple(sh))
                self._accum_grad(vnp.ones(orig_shape, dtype=_np.float32) * g_exp)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        s = self.sum(axis=axis, keepdims=keepdims)
        denom = float(self.size) / float(max(1, s.size))
        return s * (1.0 / denom)

    # --- Elementwise activations & math ---
    def relu(self) -> "Tensor":
        out_data = vnp.maximum(self.data, 0.0)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            mask = vnp.where(self.data > 0.0, 1.0, 0.0)
            self._accum_grad(g * mask)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + vnp.exp(-vnp.clip(self.data, -30.0, 30.0)))
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(g * sig * (1.0 - sig))

        return Tensor(sig, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def tanh(self) -> "Tensor":
        th = vnp.tanh(self.data)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(g * (1.0 - th * th))

        return Tensor(th, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def silu(self) -> "Tensor":
        # Single-pass fused Vulkan SPIR-V SiLU forward + backward
        from ._kernels import fused_silu_bwd, fused_silu_fwd

        out_data = fused_silu_fwd(self.data)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(fused_silu_bwd(self.data, g))

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def gelu(self) -> "Tensor":
        # Single-pass fused Vulkan SPIR-V GELU forward + backward
        from ._kernels import fused_gelu_bwd, fused_gelu_fwd

        out_data = fused_gelu_fwd(self.data)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(fused_gelu_bwd(self.data, g))

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def exp(self) -> "Tensor":
        ex = vnp.exp(self.data)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(g * ex)

        return Tensor(ex, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def log(self) -> "Tensor":
        x_safe = vnp.maximum(self.data, 1e-12)
        out_data = vnp.log(x_safe)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(g / x_safe)

        return Tensor(out_data, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def sqrt(self) -> "Tensor":
        sq = vnp.sqrt(vnp.maximum(self.data, 1e-12))
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            self._accum_grad(g / (sq * 2.0))

        return Tensor(sq, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def softmax(self, axis: int = -1) -> "Tensor":
        ax = axis + self.ndim if axis < 0 else axis
        if ax == self.ndim - 1 and self.ndim >= 1:
            from ._kernels import fused_softmax_bwd, fused_softmax_fwd

            sm = fused_softmax_fwd(self.data)
            req = _GRAD_ENABLED and self.requires_grad

            def _bw(g: vnp.ndarray) -> None:
                self._accum_grad(fused_softmax_bwd(sm, g))

            return Tensor(sm, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

        m = vnp.max(self.data, axis=axis, keepdims=True)
        ex = vnp.exp(self.data - m)
        sm = ex / vnp.sum(ex, axis=axis, keepdims=True)
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            dot = vnp.sum(g * sm, axis=axis, keepdims=True)
            self._accum_grad(sm * (g - dot))

        return Tensor(sm, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)

    def log_softmax(self, axis: int = -1) -> "Tensor":
        m = vnp.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - m
        lse = vnp.log(vnp.sum(vnp.exp(shifted), axis=axis, keepdims=True))
        lsm = shifted - lse
        req = _GRAD_ENABLED and self.requires_grad

        def _bw(g: vnp.ndarray) -> None:
            sm = vnp.exp(lsm)
            s_g = vnp.sum(g, axis=axis, keepdims=True)
            self._accum_grad(g - sm * s_g)

        return Tensor(lsm, requires_grad=req, dtype=None, _parents=(self,) if req else (), _backward=_bw if req else None)


# --- Tensor creation helpers ---
def tensor(data, *, requires_grad: bool = False, dtype=_np.float32) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad: bool = False, dtype=_np.float32) -> Tensor:
    sh = shape[0] if (len(shape) == 1 and isinstance(shape[0], (tuple, list))) else shape
    return Tensor(vnp.zeros(sh, dtype=dtype), requires_grad=requires_grad, dtype=None)


def ones(*shape, requires_grad: bool = False, dtype=_np.float32) -> Tensor:
    sh = shape[0] if (len(shape) == 1 and isinstance(shape[0], (tuple, list))) else shape
    return Tensor(vnp.ones(sh, dtype=dtype), requires_grad=requires_grad, dtype=None)


def randn(*shape, requires_grad: bool = False, seed: Optional[int] = None) -> Tensor:
    sh = shape[0] if (len(shape) == 1 and isinstance(shape[0], (tuple, list))) else shape
    rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
    return Tensor(rng.standard_normal(sh).astype(_np.float32), requires_grad=requires_grad)


def cat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Concatenate a sequence of Tensors along ``axis`` with autograd support."""
    ts = list(tensors)
    out_data = vnp.concatenate([t.data for t in ts], axis=axis)
    req = _GRAD_ENABLED and any(t.requires_grad for t in ts)
    ax = axis + out_data.ndim if axis < 0 else axis
    sizes = [t.shape[ax] for t in ts]

    def _bw(g: vnp.ndarray) -> None:
        g_np = g.to_numpy()
        off = 0
        for t, sz in zip(ts, sizes):
            if t.requires_grad:
                slc = [slice(None)] * g_np.ndim
                slc[ax] = slice(off, off + sz)
                t._accum_grad(vnp.array(_np.ascontiguousarray(g_np[tuple(slc)]), dtype=_np.float32))
            off += sz

    return Tensor(out_data, requires_grad=req, dtype=None, _parents=tuple(ts) if req else (), _backward=_bw if req else None)
