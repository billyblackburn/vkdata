"""vktorch.functional: Differentiable Vulkan GPU activations, losses, conv2d, pooling, & attention."""

from __future__ import annotations

import math
from typing import Optional

import numpy as _np
import vknumpy as vnp

from .tensor import Tensor, _to_vnp, is_grad_enabled


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis=axis)


def dropout(x: Tensor, p: float = 0.1, training: bool = True, seed: Optional[int] = None) -> Tensor:
    if not training or p <= 0.0:
        return x
    rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
    keep = (rng.uniform(0.0, 1.0, size=x.shape) >= p).astype(_np.float32) * (1.0 / (1.0 - p))
    return x * Tensor(keep, requires_grad=False)


def layer_norm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    norm = diff / (var + eps).sqrt()
    if weight is not None:
        norm = norm * weight
    if bias is not None:
        norm = norm + bias
    return norm


def embedding(indices, weight: Tensor) -> Tensor:
    """Differentiable embedding lookup ``weight[indices]`` on Vulkan GPU."""
    idx_np = _np.asarray(indices.to_numpy() if isinstance(indices, (Tensor, vnp.ndarray)) else indices, dtype=_np.int32)
    flat_idx = idx_np.ravel()
    idx_gpu = vnp.array(flat_idx, dtype=_np.int32)
    out_shape = idx_np.shape + (weight.shape[1],)
    out_data = vnp.take(weight.data, idx_gpu, axis=0).reshape(out_shape)

    req = is_grad_enabled() and weight.requires_grad

    def _bw(g: vnp.ndarray) -> None:
        g_np = g.to_numpy().reshape(-1, weight.shape[1])
        dw = _np.zeros(weight.shape, dtype=_np.float32)
        _np.add.at(dw, flat_idx, g_np)
        weight._accum_grad(vnp.array(dw, dtype=_np.float32))

    return Tensor(out_data, requires_grad=req, dtype=None, _parents=(weight,) if req else (), _backward=_bw if req else None)


def mse_loss(pred: Tensor, target) -> Tensor:
    t = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    diff = pred - t
    return (diff * diff).mean()


def binary_cross_entropy_with_logits(logits: Tensor, target) -> Tensor:
    """Numerically stable BCE-with-logits loss on Vulkan GPU."""
    t = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    # max(x, 0) - x * y + log(1 + exp(-|x|))
    x = logits.data
    y = t.data
    pos = vnp.maximum(x, 0.0)
    abs_x = vnp.maximum(x, -x)
    loss_el = pos - x * y + vnp.log(1.0 + vnp.exp(-abs_x))
    loss_val = vnp.mean(loss_el)
    req = is_grad_enabled() and logits.requires_grad

    def _bw(g: vnp.ndarray) -> None:
        sig = 1.0 / (1.0 + vnp.exp(-vnp.clip(x, -30.0, 30.0)))
        logits._accum_grad(((sig - y) * (1.0 / float(x.size))) * g)

    return Tensor(loss_val, requires_grad=req, dtype=None, _parents=(logits,) if req else (), _backward=_bw if req else None)


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Numerically stable multi-class cross-entropy loss on Vulkan GPU.

    Parameters
    ----------
    logits : Tensor of shape (N, C) or (..., C)
    targets : int32/int64 class indices of shape (N,) or (...)
    """
    tgt_np = _np.asarray(targets.to_numpy() if isinstance(targets, (Tensor, vnp.ndarray)) else targets, dtype=_np.int32).ravel()
    c_dim = logits.shape[-1]
    n_rows = logits.size // c_dim
    x2d = logits.data.contiguous().reshape(n_rows, c_dim)

    # Numerically stable log-softmax on GPU
    m = vnp.max(x2d, axis=1, keepdims=True)
    shifted = x2d - m
    ex = vnp.exp(shifted)
    sum_ex = vnp.sum(ex, axis=1, keepdims=True)
    lsm = shifted - vnp.log(sum_ex)

    # One-hot target mask on GPU
    oh_np = _np.zeros((n_rows, c_dim), dtype=_np.float32)
    oh_np[_np.arange(n_rows), tgt_np] = 1.0
    oh_gpu = vnp.array(oh_np, dtype=_np.float32)

    loss_val = -vnp.sum(lsm * oh_gpu) * (1.0 / float(n_rows))
    req = is_grad_enabled() and logits.requires_grad
    orig_shape = logits.shape

    def _bw(g: vnp.ndarray) -> None:
        probs = ex / sum_ex
        grad_2d = ((probs - oh_gpu) * (1.0 / float(n_rows))) * g
        logits._accum_grad(grad_2d.reshape(orig_shape))

    return Tensor(loss_val, requires_grad=req, dtype=None, _parents=(logits,) if req else (), _backward=_bw if req else None)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
) -> Tensor:
    """Scaled dot-product attention ``softmax((Q @ K^T) / sqrt(d_k)) @ V`` on Vulkan GPU."""
    d_k = q.shape[-1]
    scale = 1.0 / math.sqrt(float(d_k))
    scores = (q @ k.transpose(-2, -1)) * scale
    if is_causal:
        t_len = scores.shape[-1]
        mask_np = _np.triu(_np.full((t_len, t_len), -1e9, dtype=_np.float32), k=1)
        scores = scores + Tensor(mask_np, requires_grad=False)
    attn = scores.softmax(axis=-1)
    return attn @ v


def _im2col(x_np: _np.ndarray, kh: int, kw: int, stride: int, padding: int):
    n, c, h, w = x_np.shape
    if padding > 0:
        xp = _np.pad(x_np, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    else:
        xp = x_np
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1
    # Strided view (zero-copy!) -> contiguous (N*oh*ow, C*kh*kw)
    s0, s1, s2, s3 = xp.strides
    cols = _np.lib.stride_tricks.as_strided(
        xp,
        shape=(n, oh, ow, c, kh, kw),
        strides=(s0, s2 * stride, s3 * stride, s1, s2, s3),
    )
    return _np.ascontiguousarray(cols.reshape(n * oh * ow, c * kh * kw), dtype=_np.float32), oh, ow


def _col2im(dcols_np: _np.ndarray, x_shape, kh: int, kw: int, stride: int, padding: int, oh: int, ow: int) -> _np.ndarray:
    n, c, h, w = x_shape
    hp, wp = h + 2 * padding, w + 2 * padding
    dxp = _np.zeros((n, c, hp, wp), dtype=_np.float32)
    dc = dcols_np.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)  # (N, C, kh, kw, oh, ow)
    for i in range(kh):
        i_max = i + stride * oh
        for j in range(kw):
            j_max = j + stride * ow
            dxp[:, :, i:i_max:stride, j:j_max:stride] += dc[:, :, i, j, :, :]
    if padding > 0:
        return _np.ascontiguousarray(dxp[:, :, padding:-padding, padding:-padding])
    return dxp


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    """2D Convolution via ``im2col`` + Vulkan GPU GEMM (supports full autograd)."""
    n, c_in, h, w = x.shape
    c_out, c_in_w, kh, kw = weight.shape
    assert c_in == c_in_w, f"Expected {c_in_w} input channels, got {c_in}"

    cols_np, oh, ow = _im2col(x.data.to_numpy(), kh, kw, stride, padding)
    cols_gpu = vnp.array(cols_np, dtype=_np.float32)  # (N*oh*ow, C_in*kh*kw)
    w_mat = weight.data.contiguous().reshape(c_out, c_in * kh * kw)  # (C_out, K)

    out_2d = cols_gpu @ w_mat.T.contiguous()  # (N*oh*ow, C_out) on Vulkan GPU!
    if bias is not None:
        out_2d = out_2d + bias.data.reshape(1, c_out)

    # (N, oh, ow, C_out) -> (N, C_out, oh, ow)
    out_4d = out_2d.reshape(n, oh, ow, c_out).transpose((0, 3, 1, 2)).contiguous()
    req = is_grad_enabled() and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))

    parents = [x, weight] + ([bias] if bias is not None else [])

    def _bw(g: vnp.ndarray) -> None:
        # g: (N, C_out, oh, ow) -> (N*oh*ow, C_out)
        g2d = g.transpose((0, 2, 3, 1)).contiguous().reshape(n * oh * ow, c_out)
        if bias is not None and bias.requires_grad:
            bias._accum_grad(vnp.sum(g2d, axis=0))
        if weight.requires_grad:
            dw = (g2d.T.contiguous() @ cols_gpu).reshape(c_out, c_in, kh, kw)
            weight._accum_grad(dw)
        if x.requires_grad:
            dcols = (g2d @ w_mat).to_numpy()
            dx_np = _col2im(dcols, (n, c_in, h, w), kh, kw, stride, padding, oh, ow)
            x._accum_grad(vnp.array(dx_np, dtype=_np.float32))

    return Tensor(out_4d, requires_grad=req, dtype=None, _parents=tuple(parents) if req else (), _backward=_bw if req else None)


def max_pool2d(x: Tensor, kernel_size: int = 2, stride: Optional[int] = None) -> Tensor:
    """2D Max Pooling with exact argmax backward pass."""
    k = int(kernel_size)
    st = int(stride) if stride is not None else k
    x_np = x.data.to_numpy()
    n, c, h, w = x_np.shape
    oh = (h - k) // st + 1
    ow = (w - k) // st + 1

    s0, s1, s2, s3 = x_np.strides
    patches = _np.lib.stride_tricks.as_strided(
        x_np,
        shape=(n, c, oh, ow, k, k),
        strides=(s0, s1, s2 * st, s3 * st, s2, s3),
    ).reshape(n, c, oh, ow, k * k)

    argmax = _np.argmax(patches, axis=-1)
    out_np = _np.max(patches, axis=-1).astype(_np.float32)
    req = is_grad_enabled() and x.requires_grad

    def _bw(g: vnp.ndarray) -> None:
        g_np = g.to_numpy()
        dx = _np.zeros_like(x_np)
        oy_grid, ox_grid = _np.meshgrid(_np.arange(oh), _np.arange(ow), indexing="ij")
        ky = argmax // k
        kx = argmax % k
        iy = oy_grid[None, None, :, :] * st + ky
        ix = ox_grid[None, None, :, :] * st + kx
        ni = _np.arange(n)[:, None, None, None]
        ci = _np.arange(c)[None, :, None, None]
        _np.add.at(dx, (ni, ci, iy, ix), g_np)
        x._accum_grad(vnp.array(dx, dtype=_np.float32))

    return Tensor(out_np, requires_grad=req, dtype=_np.float32, _parents=(x,) if req else (), _backward=_bw if req else None)
