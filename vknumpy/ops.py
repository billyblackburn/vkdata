"""GEMM, BLAS, elementwise ufuncs, reductions, and shape ops backed by ggml-vulkan."""

from __future__ import annotations

import builtins
import math
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as _np

from ._backend import (
    NATIVE_FLOAT_DTYPES,
    NCNN_BIN_ATAN2,
    NCNN_BIN_FLOOR_DIVIDE,
    NCNN_BIN_FMOD,
    NCNN_BIN_LOGADDEXP,
    NCNN_BIN_POW,
    NCNN_BIN_REMAINDER,
    NCNN_REDUCE_MAX,
    NCNN_REDUCE_MIN,
    NCNN_REDUCE_PROD,
    NCNN_UNARY_ACOS,
    NCNN_UNARY_ASIN,
    NCNN_UNARY_ASINH,
    NCNN_UNARY_ATAN,
    NP_TO_VKN_DTYPE,
    VKN_BIN_ADD,
    VKN_BIN_DIV,
    VKN_BIN_EQ,
    VKN_BIN_GE,
    VKN_BIN_GT,
    VKN_BIN_LE,
    VKN_BIN_LT,
    VKN_BIN_MAX,
    VKN_BIN_MIN,
    VKN_BIN_MUL,
    VKN_BIN_NE,
    VKN_BIN_SUB,
    VKN_UNARY_ABS,
    VKN_UNARY_CEIL,
    VKN_UNARY_COS,
    VKN_UNARY_EXP,
    VKN_UNARY_EXPM1,
    VKN_UNARY_FLOOR,
    VKN_UNARY_GELU,
    VKN_UNARY_LOG,
    VKN_UNARY_NEG,
    VKN_UNARY_RELU,
    VKN_UNARY_ROUND,
    VKN_UNARY_SGN,
    VKN_UNARY_SIGMOID,
    VKN_UNARY_SILU,
    VKN_UNARY_SIN,
    VKN_UNARY_SQR,
    VKN_UNARY_SQRT,
    VKN_UNARY_STEP,
    VKN_UNARY_TANH,
    VKN_UNARY_TRUNC,
    check_rc,
    i32_array,
    i64_array,
    lib,
)
from .creation import asarray
from .ndarray import ndarray


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, bool, _np.number))


def _common_float_dtype(a: ndarray, b: ndarray) -> _np.dtype:
    dt = _np.result_type(a.dtype, b.dtype)
    return dt


def _broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(_np.broadcast_shapes(s1, s2))


# ======================================================================
# GEMM & BLAS (llama.cpp ggml-vulkan)
# ======================================================================

def matmul(a: Any, b: Any) -> ndarray:
    """Matrix product of two arrays using llama.cpp's ggml-vulkan GEMM."""
    xa = asarray(a)
    xb = asarray(b)

    if xa.ndim == 0 or xb.ndim == 0:
        raise ValueError("matmul: Input operand does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)")

    dt = _common_float_dtype(xa, xb)
    if dt not in NATIVE_FLOAT_DTYPES:
        return ndarray._from_numpy(_np.matmul(xa.to_numpy(), xb.to_numpy()))

    if xa.dtype != dt:
        xa = xa.astype(dt)
    if xb.dtype != dt:
        xb = xb.astype(dt)

    # 1D @ 1D -> scalar (0D ndarray)
    if xa.ndim == 1 and xb.ndim == 1:
        if xa.shape[0] != xb.shape[0]:
            raise ValueError(f"matmul: shape mismatch {xa.shape} and {xb.shape}")
        ca = xa.contiguous()
        cb = xb.contiguous()
        out = ndarray._empty_uninit((), dt)
        rc = lib().vkn_dot_1d(
            ca._buf.ptr, ca._offset,
            cb._buf.ptr, cb._offset,
            ca.shape[0],
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[dt],
        )
        check_rc(rc, "vkn_dot_1d")
        return out

    squeeze_a = False
    squeeze_b = False
    if xa.ndim == 1:
        xa = xa.reshape(1, xa.shape[0])
        squeeze_a = True
    if xb.ndim == 1:
        # Store as (1, K) with transposed_last2=True so logical shape is (K, 1) -> zero-copy!
        cb = xb.contiguous()
        xb = ndarray(cb._buf, (cb.shape[0], 1), cb.dtype, offset=cb._offset, transposed_last2=True)
        squeeze_b = True

    M, K = xa.shape[-2], xa.shape[-1]
    K2, N = xb.shape[-2], xb.shape[-1]
    if K != K2:
        raise ValueError(f"matmul: inner dimension mismatch {xa.shape} @ {xb.shape}")

    batch_shape = _broadcast_shapes(xa.shape[:-2], xb.shape[:-2])
    out_shape = batch_shape + (M, N)

    # Flatten >2 batch dims into 1 batch dim (3D total) if needed
    if len(batch_shape) > 2:
        b_flat = math.prod(batch_shape)
        xa_b = _broadcast_to_batch(xa, batch_shape).reshape(b_flat, M, K)
        xb_b = _broadcast_to_batch(xb, batch_shape).reshape(b_flat, K, N)
        res = matmul(xa_b, xb_b).reshape(out_shape)
    else:
        ca = xa.contiguous()
        b_transposed = 1 if xb._transposed_last2 else 0
        b_storage_shape = xb._storage_shape
        out = ndarray._empty_uninit(out_shape, dt)
        if out.size > 0 and K > 0:
            rc = lib().vkn_matmul(
                ca._buf.ptr, ca._offset, i64_array(ca.shape), ca.ndim,
                xb._buf.ptr, xb._offset, i64_array(b_storage_shape), len(b_storage_shape),
                b_transposed,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[dt],
            )
            check_rc(rc, "vkn_matmul")
        elif out.size > 0:
            out._buf.upload(_np.zeros(out_shape, dtype=dt))
        res = out

    if squeeze_a and squeeze_b:
        return res.reshape(())
    if squeeze_a:
        return res.reshape(batch_shape + (N,))
    if squeeze_b:
        return res.reshape(batch_shape + (M,))
    return res


def _broadcast_to_batch(x: ndarray, batch_shape: Tuple[int, ...]) -> ndarray:
    target = batch_shape + x.shape[-2:]
    if x.shape == target:
        return x
    return ndarray._from_numpy(_np.broadcast_to(x.to_numpy(), target).copy())


def dot(a: Any, b: Any) -> Any:
    """Dot product of two arrays (NumPy semantics), accelerated via ggml-vulkan."""
    xa = asarray(a)
    xb = asarray(b)

    if xa.ndim == 0 or xb.ndim == 0:
        return multiply(xa, xb)
    if xa.ndim == 1 and xb.ndim == 1:
        return matmul(xa, xb)
    if xa.ndim == 2 and xb.ndim == 2:
        return matmul(xa, xb)
    if xb.ndim == 1:
        return matmul(xa, xb)

    # General N-D dot: sum product over last axis of a and second-to-last axis of b
    # Reshape a to (-1, K) and transpose/reshape b to (K, -1), run GPU GEMM
    K = xa.shape[-1]
    if xb.shape[-2] != K:
        raise ValueError(f"shapes {xa.shape} and {xb.shape} not aligned")
    a_2d = xa.reshape(-1, K)
    perm = list(range(xb.ndim))
    perm.pop(-2)
    perm.insert(0, xb.ndim - 2)
    b_2d = xb.transpose(perm).reshape(K, -1)
    out_shape = xa.shape[:-1] + xb.shape[:-2] + (xb.shape[-1],)
    return matmul(a_2d, b_2d).reshape(out_shape)


def vdot(a: Any, b: Any) -> ndarray:
    """Return the dot product of two flattened vectors on GPU."""
    return matmul(asarray(a).ravel(), asarray(b).ravel())


def inner(a: Any, b: Any) -> ndarray:
    """Inner product of two arrays over their last axes."""
    xa = asarray(a)
    xb = asarray(b)
    if xa.ndim == 0 or xb.ndim == 0:
        return multiply(xa, xb)
    if xa.ndim == 1 and xb.ndim == 1:
        return matmul(xa, xb)
    K = xa.shape[-1]
    if xb.shape[-1] != K:
        raise ValueError(f"shapes {xa.shape} and {xb.shape} not aligned for inner")
    a_2d = xa.reshape(-1, K)
    b_2d = xb.reshape(-1, K).T
    out_shape = xa.shape[:-1] + xb.shape[:-1]
    return matmul(a_2d, b_2d).reshape(out_shape)


def outer(a: Any, b: Any) -> ndarray:
    """Compute the outer product of two vectors via ggml-vulkan GEMM."""
    xa = asarray(a).ravel()
    xb = asarray(b).ravel()
    dt = _common_float_dtype(xa, xb)
    if dt not in NATIVE_FLOAT_DTYPES:
        return ndarray._from_numpy(_np.outer(xa.to_numpy(), xb.to_numpy()))
    if xa.dtype != dt:
        xa = xa.astype(dt)
    if xb.dtype != dt:
        xb = xb.astype(dt)
    ca = xa.contiguous()
    cb = xb.contiguous()
    out = ndarray._empty_uninit((ca.shape[0], cb.shape[0]), dt)
    if out.size > 0:
        rc = lib().vkn_outer(
            ca._buf.ptr, ca._offset, ca.shape[0],
            cb._buf.ptr, cb._offset, cb.shape[0],
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[dt],
        )
        check_rc(rc, "vkn_outer")
    return out


def tensordot(a: Any, b: Any, axes: Any = 2) -> ndarray:
    """Compute tensor dot product along specified axes using GPU GEMM."""
    xa = asarray(a)
    xb = asarray(b)
    if isinstance(axes, int):
        axes_a = list(range(xa.ndim - axes, xa.ndim))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
        axes_a = [axes_a] if isinstance(axes_a, int) else list(axes_a)
        axes_b = [axes_b] if isinstance(axes_b, int) else list(axes_b)

    axes_a = [ax + xa.ndim if ax < 0 else ax for ax in axes_a]
    axes_b = [ax + xb.ndim if ax < 0 else ax for ax in axes_b]

    free_a = [i for i in range(xa.ndim) if i not in axes_a]
    free_b = [i for i in range(xb.ndim) if i not in axes_b]

    k_size = math.prod(xa.shape[i] for i in axes_a) if axes_a else 1
    k_size_b = math.prod(xb.shape[i] for i in axes_b) if axes_b else 1
    if k_size != k_size_b:
        raise ValueError("shape-mismatch for sum in tensordot")

    a_perm = free_a + axes_a
    b_perm = axes_b + free_b

    a_2d = xa.transpose(a_perm).reshape(-1, k_size)
    b_2d = xb.transpose(b_perm).reshape(k_size, -1)
    out_shape = tuple(xa.shape[i] for i in free_a) + tuple(xb.shape[i] for i in free_b)
    return matmul(a_2d, b_2d).reshape(out_shape)


def einsum(subscripts: str, *operands: Any) -> ndarray:
    """Einstein summation with fast GPU GEMM/BLAS lowering for 2-operand contractions."""
    ops = [asarray(op) for op in operands]

    if len(ops) == 2 and "->" in subscripts and "..." not in subscripts:
        lhs, out_sub = subscripts.replace(" ", "").split("->")
        parts = lhs.split(",")
        if len(parts) == 2:
            sub_a, sub_b = parts
            # Check no repeated indices within a single operand
            if len(set(sub_a)) == len(sub_a) and len(set(sub_b)) == len(sub_b) and len(set(out_sub)) == len(out_sub):
                set_a, set_b, set_out = set(sub_a), set(sub_b), set(out_sub)
                if set_out.issubset(set_a | set_b):
                    batch_idx = [c for c in out_sub if c in set_a and c in set_b]
                    contract_idx = [c for c in sub_a if c in set_b and c not in set_out]
                    free_a_idx = [c for c in sub_a if c not in set_b and c in set_out]
                    free_b_idx = [c for c in sub_b if c not in set_a and c in set_out]

                    # Only lower if no uncontracted dropped indices (pure contraction + batch)
                    if set(batch_idx + contract_idx + free_a_idx) == set_a and set(batch_idx + contract_idx + free_b_idx) == set_b:
                        a, b = ops
                        perm_a = [sub_a.index(c) for c in (batch_idx + free_a_idx + contract_idx)]
                        perm_b = [sub_b.index(c) for c in (batch_idx + contract_idx + free_b_idx)]
                        a_t = a.transpose(perm_a)
                        b_t = b.transpose(perm_b)

                        b_dims = tuple(a.shape[sub_a.index(c)] for c in batch_idx)
                        m_dim = math.prod(a.shape[sub_a.index(c)] for c in free_a_idx) if free_a_idx else 1
                        k_dim = math.prod(a.shape[sub_a.index(c)] for c in contract_idx) if contract_idx else 1
                        n_dim = math.prod(b.shape[sub_b.index(c)] for c in free_b_idx) if free_b_idx else 1

                        a_mm = a_t.reshape(b_dims + (m_dim, k_dim))
                        b_mm = b_t.reshape(b_dims + (k_dim, n_dim))
                        res = matmul(a_mm, b_mm)

                        cur_labels = batch_idx + free_a_idx + free_b_idx
                        cur_shape = (
                            b_dims
                            + tuple(a.shape[sub_a.index(c)] for c in free_a_idx)
                            + tuple(b.shape[sub_b.index(c)] for c in free_b_idx)
                        )
                        res = res.reshape(cur_shape)
                        if cur_labels != list(out_sub):
                            out_perm = [cur_labels.index(c) for c in out_sub]
                            res = res.transpose(out_perm)
                        return res

    # General fallback
    np_ops = [op.to_numpy() for op in ops]
    res_np = _np.einsum(subscripts, *np_ops)
    return ndarray._from_numpy(_np.asarray(res_np))


# ======================================================================
# Elementwise Binary Ops & Broadcasting
# ======================================================================

def _scale_bias(x: ndarray, scale: float, bias: float) -> ndarray:
    c = x.contiguous()
    out = ndarray._empty_uninit(c.shape, c.dtype)
    if c.size > 0:
        rc = lib().vkn_scale_bias(
            c._buf.ptr, c._offset,
            c.size,
            float(scale), float(bias),
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_scale_bias")
    return out


def _binary_gpu(op_code: int, np_fn: Any, a: Any, b: Any) -> ndarray:
    # Scalar fast paths (single fused ggml_scale_bias kernel on GPU!)
    if isinstance(a, ndarray) and _is_scalar(b) and a.dtype in NATIVE_FLOAT_DTYPES:
        val = float(b)
        if op_code == VKN_BIN_ADD:
            return _scale_bias(a, 1.0, val)
        if op_code == VKN_BIN_SUB:
            return _scale_bias(a, 1.0, -val)
        if op_code == VKN_BIN_MUL:
            return _scale_bias(a, val, 0.0)
        if op_code == VKN_BIN_DIV and val != 0.0:
            return _scale_bias(a, 1.0 / val, 0.0)

    if _is_scalar(a) and isinstance(b, ndarray) and b.dtype in NATIVE_FLOAT_DTYPES:
        val = float(a)
        if op_code == VKN_BIN_ADD:
            return _scale_bias(b, 1.0, val)
        if op_code == VKN_BIN_SUB:
            return _scale_bias(b, -1.0, val)
        if op_code == VKN_BIN_MUL:
            return _scale_bias(b, val, 0.0)

    xa = asarray(a)
    xb = asarray(b)
    dt = _common_float_dtype(xa, xb)
    out_shape = _broadcast_shapes(xa.shape, xb.shape)

    if dt not in NATIVE_FLOAT_DTYPES or len(out_shape) > 4:
        return ndarray._from_numpy(_np.asarray(np_fn(xa.to_numpy(), xb.to_numpy())))

    if xa.dtype != dt:
        xa = xa.astype(dt)
    if xb.dtype != dt:
        xb = xb.astype(dt)

    ca = xa.contiguous()
    cb = xb.contiguous()

    # Handle 0-D scalars as 1-element 1D tensors
    if len(out_shape) == 0:
        a_shp = (1,)
        b_shp = (1,)
        o_shp = (1,)
        ndim = 1
    else:
        ndim = len(out_shape)
        a_shp = (1,) * (ndim - ca.ndim) + ca.shape
        b_shp = (1,) * (ndim - cb.ndim) + cb.shape
        o_shp = out_shape

    out = ndarray._empty_uninit(out_shape, dt)
    if out.size > 0:
        rc = lib().vkn_binary_op(
            op_code,
            ca._buf.ptr, ca._offset, i64_array(a_shp),
            cb._buf.ptr, cb._offset, i64_array(b_shp),
            out._buf.ptr, 0, i64_array(o_shp),
            ndim,
            NP_TO_VKN_DTYPE[dt],
        )
        check_rc(rc, "vkn_binary_op")
    return out


def add(a: Any, b: Any) -> ndarray:
    return _binary_gpu(VKN_BIN_ADD, _np.add, a, b)


def subtract(a: Any, b: Any) -> ndarray:
    return _binary_gpu(VKN_BIN_SUB, _np.subtract, a, b)


def multiply(a: Any, b: Any) -> ndarray:
    return _binary_gpu(VKN_BIN_MUL, _np.multiply, a, b)


def divide(a: Any, b: Any) -> ndarray:
    return _binary_gpu(VKN_BIN_DIV, _np.divide, a, b)


true_divide = divide


def _ncnn_unary_gpu(ncnn_op: int, np_fn: Any, x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0 and lib().vkn_ncnn_available():
        c = xa.astype(_np.float32) if xa.dtype != _np.float32 else xa.contiguous()
        out = ndarray._empty_uninit(c.shape, _np.float32)
        rc = lib().vkn_ncnn_unary(
            ncnn_op,
            c._buf.ptr, c._offset,
            c.size,
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[_np.dtype(_np.float32)],
        )
        check_rc(rc, "vkn_ncnn_unary")
        return out.astype(xa.dtype) if xa.dtype != _np.float32 else out
    return ndarray._from_numpy(_np.asarray(np_fn(xa.to_numpy())))


def _ncnn_binary_gpu(ncnn_op: int, np_fn: Any, a: Any, b: Any) -> ndarray:
    xa = asarray(a)
    xb = asarray(b)
    dt = _common_float_dtype(xa, xb)
    out_shape = _broadcast_shapes(xa.shape, xb.shape)
    if dt in NATIVE_FLOAT_DTYPES and len(out_shape) <= 4 and math.prod(out_shape) > 0 and lib().vkn_ncnn_available():
        ba = broadcast_to(xa.astype(_np.float32) if xa.dtype != _np.float32 else xa, out_shape).contiguous()
        bb = broadcast_to(xb.astype(_np.float32) if xb.dtype != _np.float32 else xb, out_shape).contiguous()
        out = ndarray._empty_uninit(out_shape, _np.float32)
        rc = lib().vkn_ncnn_binary(
            ncnn_op,
            ba._buf.ptr, ba._offset,
            bb._buf.ptr, bb._offset,
            out.size,
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[_np.dtype(_np.float32)],
        )
        check_rc(rc, "vkn_ncnn_binary")
        return out.astype(dt) if dt != _np.float32 else out
    return ndarray._from_numpy(_np.asarray(np_fn(xa.to_numpy(), xb.to_numpy())))


def power(a: Any, b: Any) -> ndarray:
    if isinstance(a, ndarray) and _is_scalar(b) and a.dtype in NATIVE_FLOAT_DTYPES:
        exp_val = float(b)
        if exp_val == 2.0:
            return square(a)
        if exp_val == 0.5:
            return sqrt(a)
        if exp_val == 1.0:
            return a.copy()
        if exp_val == -1.0:
            return divide(1.0, a)
    return _ncnn_binary_gpu(NCNN_BIN_POW, _np.power, a, b)


def arctan2(y: Any, x: Any) -> ndarray:
    return _ncnn_binary_gpu(NCNN_BIN_ATAN2, _np.arctan2, y, x)


def fmod(a: Any, b: Any) -> ndarray:
    return _ncnn_binary_gpu(NCNN_BIN_FMOD, _np.fmod, a, b)


def remainder(a: Any, b: Any) -> ndarray:
    xa = asarray(a)
    xb = asarray(b)
    if _common_float_dtype(xa, xb) in NATIVE_FLOAT_DTYPES:
        # NumPy remainder uses floored division: a - floor(a / b) * b
        return subtract(xa, multiply(floor(divide(xa, xb)), xb))
    return ndarray._from_numpy(_np.asarray(_np.remainder(xa.to_numpy(), xb.to_numpy())))


mod = remainder


def floor_divide(a: Any, b: Any) -> ndarray:
    return _ncnn_binary_gpu(NCNN_BIN_FLOOR_DIVIDE, _np.floor_divide, a, b)


def logaddexp(a: Any, b: Any) -> ndarray:
    return _ncnn_binary_gpu(NCNN_BIN_LOGADDEXP, _np.logaddexp, a, b)


def arcsin(x: Any) -> ndarray:
    return _ncnn_unary_gpu(NCNN_UNARY_ASIN, _np.arcsin, x)


def arccos(x: Any) -> ndarray:
    return _ncnn_unary_gpu(NCNN_UNARY_ACOS, _np.arccos, x)


def arctan(x: Any) -> ndarray:
    return _ncnn_unary_gpu(NCNN_UNARY_ATAN, _np.arctan, x)


def arcsinh(x: Any) -> ndarray:
    return _ncnn_unary_gpu(NCNN_UNARY_ASINH, _np.arcsinh, x)


def arccosh(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return log(add(xa, sqrt(subtract(square(xa), 1.0))))
    return ndarray._from_numpy(_np.asarray(_np.arccosh(xa.to_numpy())))


def arctanh(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return _scale_bias(log(divide(add(1.0, xa), subtract(1.0, xa))), 0.5, 0.0)
    return ndarray._from_numpy(_np.asarray(_np.arctanh(xa.to_numpy())))


# ======================================================================
# Elementwise Unary Ops
# ======================================================================

def _unary_gpu(op_code: int, np_fn: Any, x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype not in NATIVE_FLOAT_DTYPES:
        return ndarray._from_numpy(_np.asarray(np_fn(xa.to_numpy())))
    c = xa.contiguous()
    out = ndarray._empty_uninit(c.shape, c.dtype)
    if c.size > 0:
        rc = lib().vkn_unary_op(
            op_code,
            c._buf.ptr, c._offset,
            c.size,
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_unary_op")
    return out


def negative(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_NEG, _np.negative, x)


def abs(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_ABS, _np.abs, x)


absolute = abs


def square(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_SQR, _np.square, x)


def sqrt(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_SQRT, _np.sqrt, x)


def exp(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_EXP, _np.exp, x)


def exp2(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return exp(_scale_bias(xa, math.log(2.0), 0.0))
    return ndarray._from_numpy(_np.asarray(_np.exp2(xa.to_numpy())))


def log(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_LOG, _np.log, x)


def log2(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return _scale_bias(log(xa), 1.0 / math.log(2.0), 0.0)
    return ndarray._from_numpy(_np.asarray(_np.log2(xa.to_numpy())))


def log10(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return _scale_bias(log(xa), 1.0 / math.log(10.0), 0.0)
    return ndarray._from_numpy(_np.asarray(_np.log10(xa.to_numpy())))


def sin(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_SIN, _np.sin, x)


def cos(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_COS, _np.cos, x)


def tan(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return divide(sin(xa), cos(xa))
    return ndarray._from_numpy(_np.asarray(_np.tan(xa.to_numpy())))


def tanh(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_TANH, _np.tanh, x)


def relu(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_RELU, lambda a: _np.maximum(a, 0), x)


def sigmoid(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_SIGMOID, lambda a: 1.0 / (1.0 + _np.exp(-a)), x)


def silu(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_SILU, lambda a: a / (1.0 + _np.exp(-a)), x)


def gelu(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_GELU, lambda a: 0.5 * a * (1.0 + _np.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * a**3))), x)


def floor(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_FLOOR, _np.floor, x)


def ceil(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_CEIL, _np.ceil, x)


def round(x: Any, decimals: int = 0) -> ndarray:
    if decimals == 0:
        return _unary_gpu(VKN_UNARY_ROUND, _np.round, x)
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        scale = 10.0 ** decimals
        return _scale_bias(_unary_gpu(VKN_UNARY_ROUND, _np.round, _scale_bias(xa, scale, 0.0)), 1.0 / scale, 0.0)
    return ndarray._from_numpy(_np.asarray(_np.round(xa.to_numpy(), decimals=decimals)))


rint = round


def trunc(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_TRUNC, _np.trunc, x)


def sign(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_SGN, _np.sign, x)


def expm1(x: Any) -> ndarray:
    return _unary_gpu(VKN_UNARY_EXPM1, _np.expm1, x)


def log1p(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return log(_scale_bias(xa, 1.0, 1.0))
    return ndarray._from_numpy(_np.asarray(_np.log1p(xa.to_numpy())))


def sinh(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return _scale_bias(subtract(exp(xa), exp(negative(xa))), 0.5, 0.0)
    return ndarray._from_numpy(_np.asarray(_np.sinh(xa.to_numpy())))


def cosh(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        return _scale_bias(add(exp(xa), exp(negative(xa))), 0.5, 0.0)
    return ndarray._from_numpy(_np.asarray(_np.cosh(xa.to_numpy())))


def reciprocal(x: Any) -> ndarray:
    return divide(1.0, asarray(x))


def cbrt(x: Any) -> ndarray:
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES:
        # sign(x) * exp(log(abs(x)) / 3) with exact 0 handling
        s = sign(xa)
        mag = exp(_scale_bias(log(add(abs(xa), 1e-38)), 1.0 / 3.0, 0.0))
        return multiply(s, mag)
    return ndarray._from_numpy(_np.asarray(_np.cbrt(xa.to_numpy())))


def heaviside(x1: Any, x2: Any) -> ndarray:
    xa = asarray(x1)
    if xa.dtype in NATIVE_FLOAT_DTYPES and _is_scalar(x2) and float(x2) == 0.0:
        return _unary_gpu(VKN_UNARY_STEP, lambda a: _np.heaviside(a, 0.0), xa)
    xb = asarray(x2)
    return ndarray._from_numpy(_np.asarray(_np.heaviside(xa.to_numpy(), xb.to_numpy())))


def clip(a: Any, a_min: Any, a_max: Any) -> ndarray:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and (_is_scalar(a_min) or a_min is None) and (_is_scalar(a_max) or a_max is None):
        lo = float("-inf") if a_min is None else float(a_min)
        hi = float("inf") if a_max is None else float(a_max)
        c = xa.contiguous()
        out = ndarray._empty_uninit(c.shape, c.dtype)
        if c.size > 0:
            rc = lib().vkn_clamp(
                c._buf.ptr, c._offset,
                c.size,
                lo, hi,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_clamp")
        return out
    lo_np = a_min.to_numpy() if isinstance(a_min, ndarray) else a_min
    hi_np = a_max.to_numpy() if isinstance(a_max, ndarray) else a_max
    return ndarray._from_numpy(_np.asarray(_np.clip(xa.to_numpy(), lo_np, hi_np)))


def maximum(a: Any, b: Any) -> ndarray:
    if isinstance(a, ndarray) and _is_scalar(b) and a.dtype in NATIVE_FLOAT_DTYPES:
        return clip(a, float(b), None)
    if _is_scalar(a) and isinstance(b, ndarray) and b.dtype in NATIVE_FLOAT_DTYPES:
        return clip(b, float(a), None)
    return _binary_gpu(VKN_BIN_MAX, _np.maximum, a, b)


def minimum(a: Any, b: Any) -> ndarray:
    if isinstance(a, ndarray) and _is_scalar(b) and a.dtype in NATIVE_FLOAT_DTYPES:
        return clip(a, None, float(b))
    if _is_scalar(a) and isinstance(b, ndarray) and b.dtype in NATIVE_FLOAT_DTYPES:
        return clip(b, None, float(a))
    return _binary_gpu(VKN_BIN_MIN, _np.minimum, a, b)


def _compare_gpu(op_code: int, np_fn: Any, a: Any, b: Any) -> ndarray:
    xa = asarray(a)
    xb = asarray(b)
    dt = _common_float_dtype(xa, xb)
    out_shape = _broadcast_shapes(xa.shape, xb.shape)
    if dt in NATIVE_FLOAT_DTYPES and len(out_shape) <= 4:
        mask_f = _binary_gpu(op_code, np_fn, xa, xb)
        return mask_f.astype(_np.bool_)
    return ndarray._from_numpy(_np.asarray(np_fn(xa.to_numpy(), xb.to_numpy())))


def greater(a: Any, b: Any) -> ndarray:
    return _compare_gpu(VKN_BIN_GT, _np.greater, a, b)


def less(a: Any, b: Any) -> ndarray:
    return _compare_gpu(VKN_BIN_LT, _np.less, a, b)


def greater_equal(a: Any, b: Any) -> ndarray:
    return _compare_gpu(VKN_BIN_GE, _np.greater_equal, a, b)


def less_equal(a: Any, b: Any) -> ndarray:
    return _compare_gpu(VKN_BIN_LE, _np.less_equal, a, b)


def equal(a: Any, b: Any) -> ndarray:
    return _compare_gpu(VKN_BIN_EQ, _np.equal, a, b)


def not_equal(a: Any, b: Any) -> ndarray:
    return _compare_gpu(VKN_BIN_NE, _np.not_equal, a, b)


def logical_not(x: Any) -> ndarray:
    xa = asarray(x)
    return equal(xa.astype(_np.float32), 0.0)


def logical_and(a: Any, b: Any) -> ndarray:
    xa = not_equal(asarray(a).astype(_np.float32), 0.0).astype(_np.float32)
    xb = not_equal(asarray(b).astype(_np.float32), 0.0).astype(_np.float32)
    return greater(multiply(xa, xb), 0.5)


def logical_or(a: Any, b: Any) -> ndarray:
    xa = not_equal(asarray(a).astype(_np.float32), 0.0).astype(_np.float32)
    xb = not_equal(asarray(b).astype(_np.float32), 0.0).astype(_np.float32)
    return greater(add(xa, xb), 0.5)


def logical_xor(a: Any, b: Any) -> ndarray:
    xa = not_equal(asarray(a).astype(_np.float32), 0.0).astype(_np.float32)
    xb = not_equal(asarray(b).astype(_np.float32), 0.0).astype(_np.float32)
    return not_equal(xa, xb)


def count_nonzero(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and (axis is None or isinstance(axis, int)):
        nz = _binary_gpu(VKN_BIN_NE, _np.not_equal, xa, 0.0)
        return sum(nz, axis=axis, keepdims=keepdims).astype(_np.int64)
    return ndarray._from_numpy(_np.asarray(_np.count_nonzero(xa.to_numpy(), axis=axis, keepdims=keepdims)))


def isclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> ndarray:
    xa = asarray(a)
    xb = asarray(b)
    if _common_float_dtype(xa, xb) in NATIVE_FLOAT_DTYPES:
        return less_equal(abs(subtract(xa, xb)), add(atol, multiply(rtol, abs(xb))))
    return ndarray._from_numpy(_np.asarray(_np.isclose(xa.to_numpy(), xb.to_numpy(), rtol=rtol, atol=atol)))


def array_equal(a1: Any, a2: Any) -> bool:
    xa, xb = asarray(a1), asarray(a2)
    if xa.shape != xb.shape:
        return False
    if xa.size == 0:
        return True
    return bool(all(equal(xa, xb)).item())


def where(condition: Any, x: Any = None, y: Any = None) -> Any:
    if x is None and y is None:
        cond = condition.to_numpy() if isinstance(condition, ndarray) else _np.asarray(condition)
        return tuple(ndarray._from_numpy(idx) for idx in _np.where(cond))

    xc = asarray(condition)
    xx = asarray(x)
    xy = asarray(y)
    dt = _common_float_dtype(xx, xy)
    out_shape = _broadcast_shapes(_broadcast_shapes(xc.shape, xx.shape), xy.shape)

    if dt in NATIVE_FLOAT_DTYPES and len(out_shape) <= 4:
        cc = (xc.astype(_np.float32) if xc.dtype != _np.float32 else xc).contiguous()
        cx = (xx.astype(dt) if xx.dtype != dt else xx).contiguous()
        cy = (xy.astype(dt) if xy.dtype != dt else xy).contiguous()

        if len(out_shape) == 0:
            c_shp = x_shp = y_shp = o_shp = (1,)
            ndim = 1
        else:
            ndim = len(out_shape)
            c_shp = (1,) * (ndim - cc.ndim) + cc.shape
            x_shp = (1,) * (ndim - cx.ndim) + cx.shape
            y_shp = (1,) * (ndim - cy.ndim) + cy.shape
            o_shp = out_shape

        out = ndarray._empty_uninit(out_shape, dt)
        if out.size > 0:
            rc = lib().vkn_where(
                cc._buf.ptr, cc._offset, i64_array(c_shp),
                cx._buf.ptr, cx._offset, i64_array(x_shp),
                cy._buf.ptr, cy._offset, i64_array(y_shp),
                out._buf.ptr, 0, i64_array(o_shp),
                ndim,
                NP_TO_VKN_DTYPE[dt],
            )
            check_rc(rc, "vkn_where")
        return out

    return ndarray._from_numpy(_np.asarray(_np.where(xc.to_numpy(), xx.to_numpy(), xy.to_numpy())))


# ======================================================================
# Reductions
# ======================================================================

def sum(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        if axis is None:
            c = xa.contiguous()
            out_shape = (1,) * xa.ndim if keepdims else ()
            out = ndarray._empty_uninit(out_shape, c.dtype)
            rc = lib().vkn_sum_all(
                c._buf.ptr, c._offset,
                c.size,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_sum_all")
            return out

        if isinstance(axis, int):
            ax = axis + xa.ndim if axis < 0 else axis
            if ax < 0 or ax >= xa.ndim:
                raise ValueError(f"axis {axis} out of bounds")
            if ax != xa.ndim - 1:
                perm = [i for i in range(xa.ndim) if i != ax] + [ax]
                xa = xa.transpose(perm)
            c = xa.contiguous()
            cols = c.shape[-1]
            rows = c.size // cols
            reduced_shape = c.shape[:-1]
            out_shape = (
                (c.shape[:-1] + (1,)) if (keepdims and ax == xa.ndim - 1) else reduced_shape
            )
            out = ndarray._empty_uninit(reduced_shape, c.dtype)
            rc = lib().vkn_sum_rows(
                c._buf.ptr, c._offset,
                rows, cols,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_sum_rows")
            if keepdims:
                kshape = list(reduced_shape)
                kshape.insert(ax, 1)
                return out.reshape(tuple(kshape))
            return out

    return ndarray._from_numpy(_np.asarray(_np.sum(xa.to_numpy(), axis=axis, keepdims=keepdims)))


def mean(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        if axis is None:
            s = sum(xa, axis=None, keepdims=keepdims)
            return _scale_bias(s, 1.0 / xa.size, 0.0)
        if isinstance(axis, int):
            ax = axis + xa.ndim if axis < 0 else axis
            if ax < 0 or ax >= xa.ndim:
                raise ValueError(f"axis {axis} out of bounds")
            if ax != xa.ndim - 1:
                perm = [i for i in range(xa.ndim) if i != ax] + [ax]
                xa = xa.transpose(perm)
            c = xa.contiguous()
            cols = c.shape[-1]
            rows = c.size // cols
            reduced_shape = c.shape[:-1]
            out = ndarray._empty_uninit(reduced_shape, c.dtype)
            rc = lib().vkn_mean_rows(
                c._buf.ptr, c._offset,
                rows, cols,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_mean_rows")
            if keepdims:
                kshape = list(reduced_shape)
                kshape.insert(ax, 1)
                return out.reshape(tuple(kshape))
            return out

    return ndarray._from_numpy(_np.asarray(_np.mean(xa.to_numpy(), axis=axis, keepdims=keepdims)))


def var(a: Any, axis: Any = None, ddof: int = 0, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0 and (axis is None or isinstance(axis, int)):
        m = mean(xa, axis=axis, keepdims=True)
        diff_sq = square(subtract(xa, m))
        n = xa.size if axis is None else xa.shape[axis]
        s = sum(diff_sq, axis=axis, keepdims=keepdims)
        return _scale_bias(s, 1.0 / builtins.max(1, n - ddof), 0.0)
    return ndarray._from_numpy(_np.asarray(_np.var(xa.to_numpy(), axis=axis, ddof=ddof, keepdims=keepdims)))


def std(a: Any, axis: Any = None, ddof: int = 0, keepdims: bool = False) -> ndarray:
    return sqrt(var(a, axis=axis, ddof=ddof, keepdims=keepdims))


def argmax(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        if axis is None:
            c = xa.contiguous()
            out = ndarray._empty_uninit((1,) * xa.ndim if keepdims else (), _np.int32)
            rc = lib().vkn_argmax_rows(
                c._buf.ptr, c._offset,
                1, c.size,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_argmax_rows")
            return out
        if isinstance(axis, int):
            ax = axis + xa.ndim if axis < 0 else axis
            if ax != xa.ndim - 1:
                perm = [i for i in range(xa.ndim) if i != ax] + [ax]
                xa = xa.transpose(perm)
            c = xa.contiguous()
            cols = c.shape[-1]
            rows = c.size // cols
            reduced_shape = c.shape[:-1]
            out = ndarray._empty_uninit(reduced_shape, _np.int32)
            rc = lib().vkn_argmax_rows(
                c._buf.ptr, c._offset,
                rows, cols,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_argmax_rows")
            if keepdims:
                kshape = list(reduced_shape)
                kshape.insert(ax, 1)
                return out.reshape(tuple(kshape))
            return out
    return ndarray._from_numpy(_np.asarray(_np.argmax(xa.to_numpy(), axis=axis, keepdims=keepdims)))


def argmin(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        return argmax(negative(xa), axis=axis, keepdims=keepdims)
    return ndarray._from_numpy(_np.asarray(_np.argmin(xa.to_numpy(), axis=axis, keepdims=keepdims)))


def _ncnn_reduce(ncnn_reduce_op: int, np_fn: Any, a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0 and lib().vkn_ncnn_available():
        if axis is None:
            c = xa.astype(_np.float32) if xa.dtype != _np.float32 else xa.contiguous()
            out_shape = (1,) * xa.ndim if keepdims else ()
            out = ndarray._empty_uninit(out_shape, _np.float32)
            rc = lib().vkn_ncnn_reduce_rows(
                ncnn_reduce_op,
                c._buf.ptr, c._offset,
                1, c.size,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[_np.dtype(_np.float32)],
            )
            check_rc(rc, "vkn_ncnn_reduce_rows")
            return out.astype(xa.dtype) if xa.dtype != _np.float32 else out

        if isinstance(axis, int):
            ax = axis + xa.ndim if axis < 0 else axis
            if ax < 0 or ax >= xa.ndim:
                raise ValueError(f"axis {axis} out of bounds")
            orig_ndim = xa.ndim
            if ax != orig_ndim - 1:
                perm = [i for i in range(orig_ndim) if i != ax] + [ax]
                xa = xa.transpose(perm)
            c = xa.astype(_np.float32) if xa.dtype != _np.float32 else xa.contiguous()
            cols = c.shape[-1]
            rows = c.size // cols
            reduced_shape = c.shape[:-1]
            out = ndarray._empty_uninit(reduced_shape, _np.float32)
            rc = lib().vkn_ncnn_reduce_rows(
                ncnn_reduce_op,
                c._buf.ptr, c._offset,
                rows, cols,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[_np.dtype(_np.float32)],
            )
            check_rc(rc, "vkn_ncnn_reduce_rows")
            if xa.dtype != _np.float32:
                out = out.astype(xa.dtype)
            if keepdims:
                kshape = list(reduced_shape)
                kshape.insert(ax, 1)
                return out.reshape(tuple(kshape))
            return out

    return ndarray._from_numpy(_np.asarray(np_fn(xa.to_numpy(), axis=axis, keepdims=keepdims)))


def max(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    return _ncnn_reduce(NCNN_REDUCE_MAX, _np.max, a, axis=axis, keepdims=keepdims)


def min(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    return _ncnn_reduce(NCNN_REDUCE_MIN, _np.min, a, axis=axis, keepdims=keepdims)


def prod(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    return _ncnn_reduce(NCNN_REDUCE_PROD, _np.prod, a, axis=axis, keepdims=keepdims)


def all(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.size > 0 and (axis is None or isinstance(axis, int)):
        nz = _binary_gpu(VKN_BIN_NE, _np.not_equal, xa.astype(_np.float32), 0.0)
        n = xa.size if axis is None else xa.shape[axis]
        return equal(sum(nz, axis=axis, keepdims=keepdims), float(n))
    return ndarray._from_numpy(_np.asarray(_np.all(xa.to_numpy(), axis=axis, keepdims=keepdims)))


def any(a: Any, axis: Any = None, keepdims: bool = False) -> ndarray:
    xa = asarray(a)
    if xa.size > 0 and (axis is None or isinstance(axis, int)):
        nz = _binary_gpu(VKN_BIN_NE, _np.not_equal, xa.astype(_np.float32), 0.0)
        return greater(sum(nz, axis=axis, keepdims=keepdims), 0.5)
    return ndarray._from_numpy(_np.asarray(_np.any(xa.to_numpy(), axis=axis, keepdims=keepdims)))


# ======================================================================
# Shape Manipulation
# ======================================================================

def reshape(a: Any, newshape: Any) -> ndarray:
    return asarray(a).reshape(newshape)


def transpose(a: Any, axes: Any = None) -> ndarray:
    return asarray(a).transpose(axes)


def swapaxes(a: Any, axis1: int, axis2: int) -> ndarray:
    return asarray(a).swapaxes(axis1, axis2)


def moveaxis(a: Any, source: int, destination: int) -> ndarray:
    xa = asarray(a)
    src = source + xa.ndim if source < 0 else source
    dst = destination + xa.ndim if destination < 0 else destination
    order = [i for i in range(xa.ndim) if i != src]
    order.insert(dst, src)
    return xa.transpose(order)


def squeeze(a: Any, axis: Any = None) -> ndarray:
    return asarray(a).squeeze(axis=axis)


def expand_dims(a: Any, axis: int) -> ndarray:
    xa = asarray(a)
    ax = axis + xa.ndim + 1 if axis < 0 else axis
    new_shape = list(xa.shape)
    new_shape.insert(ax, 1)
    return xa.reshape(tuple(new_shape))


def _concat_two_gpu(a: ndarray, b: ndarray, axis: int) -> ndarray:
    ca = a.contiguous()
    cb = b.contiguous()
    out_shape = list(ca.shape)
    out_shape[axis] += cb.shape[axis]
    out = ndarray._empty_uninit(tuple(out_shape), ca.dtype)
    if out.size > 0:
        rc = lib().vkn_concat(
            ca._buf.ptr, ca._offset, i64_array(ca.shape),
            cb._buf.ptr, cb._offset, i64_array(cb.shape),
            ca.ndim, axis,
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[ca.dtype],
        )
        check_rc(rc, "vkn_concat")
    return out


def concatenate(arrays: Sequence[Any], axis: int = 0) -> ndarray:
    arrs = [asarray(x) for x in arrays]
    if not arrs:
        raise ValueError("need at least one array to concatenate")
    if len(arrs) == 1:
        return arrs[0].copy()

    dt = _np.result_type(*(a.dtype for a in arrs))
    ndim = arrs[0].ndim
    ax = axis + ndim if axis < 0 else axis

    if dt in NP_TO_VKN_DTYPE and 1 <= ndim <= 4 and 0 <= ax < ndim:
        casted = [a.astype(dt) if a.dtype != dt else a for a in arrs]
        if builtins.all(a.ndim == ndim and a.size > 0 for a in casted):
            acc = casted[0]
            for nxt in casted[1:]:
                acc = _concat_two_gpu(acc, nxt, ax)
            return acc

    np_arrs = [x.to_numpy() for x in arrs]
    return ndarray._from_numpy(_np.concatenate(np_arrs, axis=axis))


def stack(arrays: Sequence[Any], axis: int = 0) -> ndarray:
    arrs = [asarray(x) for x in arrays]
    if not arrs:
        raise ValueError("need at least one array to stack")
    expanded = [expand_dims(a, axis=axis) for a in arrs]
    return concatenate(expanded, axis=axis)


def vstack(tup: Sequence[Any]) -> ndarray:
    arrs = [asarray(x) for x in tup]
    prepped = [a.reshape(1, a.shape[0]) if a.ndim == 1 else a for a in arrs]
    return concatenate(prepped, axis=0)


def hstack(tup: Sequence[Any]) -> ndarray:
    arrs = [asarray(x) for x in tup]
    if arrs and arrs[0].ndim == 1:
        return concatenate(arrs, axis=0)
    return concatenate(arrs, axis=1)


def dstack(tup: Sequence[Any]) -> ndarray:
    arrs = [asarray(x) for x in tup]
    prepped = []
    for a in arrs:
        if a.ndim == 1:
            prepped.append(a.reshape(1, a.shape[0], 1))
        elif a.ndim == 2:
            prepped.append(a.reshape(a.shape[0], a.shape[1], 1))
        else:
            prepped.append(a)
    return concatenate(prepped, axis=2)


def pad(array: Any, pad_width: Any, mode: str = "constant", **kwargs: Any) -> ndarray:
    """Pad an array; uses GPU ggml_pad_ext for zero constant padding up to 4D."""
    xa = asarray(array)
    constant_values = kwargs.get("constant_values", 0)
    if (
        mode == "constant"
        and constant_values == 0
        and xa.dtype in NATIVE_FLOAT_DTYPES
        and 1 <= xa.ndim <= 4
        and xa.size > 0
    ):
        if isinstance(pad_width, int):
            before = [pad_width] * xa.ndim
            after = [pad_width] * xa.ndim
        elif isinstance(pad_width, (tuple, list)) and len(pad_width) == 2 and isinstance(pad_width[0], int):
            before = [int(pad_width[0])] * xa.ndim
            after = [int(pad_width[1])] * xa.ndim
        else:
            before = [int(p[0]) for p in pad_width]
            after = [int(p[1]) for p in pad_width]

        if len(before) == xa.ndim and builtins.all(b >= 0 and a >= 0 for b, a in zip(before, after)):
            c = xa.contiguous()
            out_shape = tuple(c.shape[i] + before[i] + after[i] for i in range(c.ndim))
            out = ndarray._empty_uninit(out_shape, c.dtype)
            rc = lib().vkn_pad(
                c._buf.ptr, c._offset,
                i64_array(c.shape), c.ndim,
                i32_array(before), i32_array(after),
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_pad")
            return out

    return ndarray._from_numpy(_np.pad(xa.to_numpy(), pad_width, mode=mode, **kwargs))


def roll(a: Any, shift: Union[int, Sequence[int]], axis: Optional[Union[int, Sequence[int]]] = None) -> ndarray:
    """Roll array elements along given axis/axes on Vulkan GPU via ggml_roll."""
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and 1 <= xa.ndim <= 4 and xa.size > 0:
        if axis is None:
            flat = xa.ravel()
            c = flat.contiguous()
            sh = int(shift) % c.size
            out = ndarray._empty_uninit(c.shape, c.dtype)
            rc = lib().vkn_roll(
                c._buf.ptr, c._offset,
                i64_array(c.shape), 1,
                i32_array([sh]),
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_roll")
            return out.reshape(xa.shape)

        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        shifts_in = (shift,) * len(axes) if isinstance(shift, int) else tuple(shift)
        shifts_per_dim = [0] * xa.ndim
        for ax, sh in zip(axes, shifts_in):
            ax_n = ax + xa.ndim if ax < 0 else ax
            shifts_per_dim[ax_n] = (shifts_per_dim[ax_n] + int(sh)) % xa.shape[ax_n]
        c = xa.contiguous()
        out = ndarray._empty_uninit(c.shape, c.dtype)
        rc = lib().vkn_roll(
            c._buf.ptr, c._offset,
            i64_array(c.shape), c.ndim,
            i32_array(shifts_per_dim),
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_roll")
        return out

    return ndarray._from_numpy(_np.roll(xa.to_numpy(), shift, axis=axis))


def tile(A: Any, reps: Union[int, Sequence[int]]) -> ndarray:
    """Construct an array by repeating A on Vulkan GPU via ggml_repeat."""
    xa = asarray(A)
    r = (reps,) if isinstance(reps, int) else tuple(int(x) for x in reps)
    ndim = builtins.max(xa.ndim, len(r))
    a_shp = (1,) * (ndim - xa.ndim) + xa.shape
    r_pad = (1,) * (ndim - len(r)) + r
    out_shape = tuple(a_shp[i] * r_pad[i] for i in range(ndim))

    if xa.dtype in NATIVE_FLOAT_DTYPES and 1 <= ndim <= 4 and math.prod(out_shape) > 0:
        c = xa.contiguous()
        out = ndarray._empty_uninit(out_shape, c.dtype)
        rc = lib().vkn_repeat(
            c._buf.ptr, c._offset, i64_array(a_shp),
            out._buf.ptr, 0, i64_array(out_shape),
            ndim,
            NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_repeat")
        return out

    return ndarray._from_numpy(_np.tile(xa.to_numpy(), reps))


def broadcast_to(array: Any, shape: Union[int, Sequence[int]]) -> ndarray:
    """Broadcast an array to a new shape on Vulkan GPU via ggml_repeat."""
    xa = asarray(array)
    target = (shape,) if isinstance(shape, int) else tuple(int(d) for d in shape)
    if xa.shape == target:
        return xa.copy()
    ndim = len(target)
    if xa.dtype in NATIVE_FLOAT_DTYPES and 1 <= ndim <= 4 and xa.ndim <= ndim and math.prod(target) > 0:
        a_shp = (1,) * (ndim - xa.ndim) + xa.shape
        if builtins.all(a_shp[i] == target[i] or a_shp[i] == 1 for i in range(ndim)):
            c = xa.contiguous()
            out = ndarray._empty_uninit(target, c.dtype)
            rc = lib().vkn_repeat(
                c._buf.ptr, c._offset, i64_array(a_shp),
                out._buf.ptr, 0, i64_array(target),
                ndim,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_repeat")
            return out
    return ndarray._from_numpy(_np.broadcast_to(xa.to_numpy(), target).copy())


def split(ary: Any, indices_or_sections: Any, axis: int = 0) -> list[ndarray]:
    xa = asarray(ary)
    parts = _np.split(xa.to_numpy(), indices_or_sections, axis=axis)
    return [ndarray._from_numpy(p) for p in parts]


def array_split(ary: Any, indices_or_sections: Any, axis: int = 0) -> list[ndarray]:
    xa = asarray(ary)
    parts = _np.array_split(xa.to_numpy(), indices_or_sections, axis=axis)
    return [ndarray._from_numpy(p) for p in parts]


def hsplit(ary: Any, indices_or_sections: Any) -> list[ndarray]:
    xa = asarray(ary)
    return split(xa, indices_or_sections, axis=0 if xa.ndim == 1 else 1)


def vsplit(ary: Any, indices_or_sections: Any) -> list[ndarray]:
    return split(ary, indices_or_sections, axis=0)


def flip(m: Any, axis: Optional[Union[int, Sequence[int]]] = None) -> ndarray:
    return ndarray._from_numpy(_np.flip(asarray(m).to_numpy(), axis=axis).copy())


# ======================================================================
# Gather, Scans & Sorting (ggml-vulkan)
# ======================================================================

def take(a: Any, indices: Any, axis: Optional[int] = None) -> ndarray:
    """Take elements from an array along an axis on GPU via ggml_get_rows."""
    xa = asarray(a)
    idx = asarray(indices, dtype=_np.int32)

    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0 and idx.size > 0:
        if axis is None:
            flat = xa.ravel().contiguous()
            idx_c = idx.ravel().contiguous()
            out = ndarray._empty_uninit((idx_c.size, 1), flat.dtype)
            rc = lib().vkn_get_rows(
                flat._buf.ptr, flat._offset, flat.size, 1,
                idx_c._buf.ptr, idx_c._offset, idx_c.size,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[flat.dtype],
            )
            check_rc(rc, "vkn_get_rows")
            return out.reshape(idx.shape)

        ax = axis + xa.ndim if axis < 0 else axis
        if ax == 0:
            c = xa.contiguous()
            rows = c.shape[0]
            cols = c.size // rows
            idx_c = idx.ravel().contiguous()
            out = ndarray._empty_uninit((idx_c.size, cols), c.dtype)
            rc = lib().vkn_get_rows(
                c._buf.ptr, c._offset, rows, cols,
                idx_c._buf.ptr, idx_c._offset, idx_c.size,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_get_rows")
            return out.reshape(idx.shape + c.shape[1:])

    return ndarray._from_numpy(_np.take(xa.to_numpy(), idx.to_numpy(), axis=axis))


def cumsum(a: Any, axis: Optional[int] = None) -> ndarray:
    """Cumulative sum on Vulkan GPU via ggml_cumsum."""
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        if axis is None:
            c = xa.ravel().contiguous()
            out = ndarray._empty_uninit(c.shape, c.dtype)
            rc = lib().vkn_cumsum_rows(
                c._buf.ptr, c._offset, 1, c.size,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_cumsum_rows")
            return out

        ax = axis + xa.ndim if axis < 0 else axis
        if ax < 0 or ax >= xa.ndim:
            raise ValueError(f"axis {axis} out of bounds")
        perm = None
        if ax != xa.ndim - 1:
            perm = [i for i in range(xa.ndim) if i != ax] + [ax]
            xa = xa.transpose(perm)
        c = xa.contiguous()
        cols = c.shape[-1]
        rows = c.size // cols
        out = ndarray._empty_uninit(c.shape, c.dtype)
        rc = lib().vkn_cumsum_rows(
            c._buf.ptr, c._offset, rows, cols,
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_cumsum_rows")
        if perm is not None:
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            out = out.transpose(inv_perm)
        return out

    return ndarray._from_numpy(_np.asarray(_np.cumsum(xa.to_numpy(), axis=axis)))


def argsort(a: Any, axis: int = -1, descending: bool = False) -> ndarray:
    """Return the indices that would sort an array on Vulkan GPU via ggml_argsort."""
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        if axis is None:
            c = xa.ravel().contiguous()
            out = ndarray._empty_uninit(c.shape, _np.int32)
            rc = lib().vkn_argsort_rows(
                c._buf.ptr, c._offset, 1, c.size,
                1 if descending else 0,
                out._buf.ptr, 0,
                NP_TO_VKN_DTYPE[c.dtype],
            )
            check_rc(rc, "vkn_argsort_rows")
            return out

        ax = axis + xa.ndim if axis < 0 else axis
        perm = None
        if ax != xa.ndim - 1:
            perm = [i for i in range(xa.ndim) if i != ax] + [ax]
            xa = xa.transpose(perm)
        c = xa.contiguous()
        cols = c.shape[-1]
        rows = c.size // cols
        out = ndarray._empty_uninit(c.shape, _np.int32)
        rc = lib().vkn_argsort_rows(
            c._buf.ptr, c._offset, rows, cols,
            1 if descending else 0,
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_argsort_rows")
        if perm is not None:
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            out = out.transpose(inv_perm)
        return out

    res = _np.argsort(xa.to_numpy(), axis=axis)
    if descending:
        res = _np.flip(res, axis=axis if axis is not None else 0).copy()
    return ndarray._from_numpy(res)


def sort(a: Any, axis: int = -1, descending: bool = False) -> ndarray:
    """Return a sorted copy of an array (uses GPU argsort + gather for 1D/last-axis)."""
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0 and (xa.ndim == 1 or axis is None):
        flat = xa.ravel()
        idx = argsort(flat, axis=0, descending=descending)
        return take(flat, idx)
    res = _np.sort(xa.to_numpy(), axis=axis)
    if descending:
        res = _np.flip(res, axis=axis if axis is not None else 0).copy()
    return ndarray._from_numpy(res)


def top_k(a: Any, k: int, axis: int = -1) -> ndarray:
    """Return indices of the top-k largest elements along axis on GPU via ggml_top_k."""
    xa = asarray(a)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        ax = axis + xa.ndim if axis < 0 else axis
        perm = None
        if ax != xa.ndim - 1:
            perm = [i for i in range(xa.ndim) if i != ax] + [ax]
            xa = xa.transpose(perm)
        c = xa.contiguous()
        cols = c.shape[-1]
        rows = c.size // cols
        out_shape = c.shape[:-1] + (k,)
        out = ndarray._empty_uninit(out_shape, _np.int32)
        rc = lib().vkn_top_k_rows(
            c._buf.ptr, c._offset, rows, cols,
            int(k),
            out._buf.ptr, 0,
            NP_TO_VKN_DTYPE[c.dtype],
        )
        check_rc(rc, "vkn_top_k_rows")
        if perm is not None:
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            out = out.transpose(inv_perm)
        return out
    return argsort(xa, axis=axis, descending=True)[..., :k]


def copy(a: Any) -> ndarray:
    return asarray(a).copy()


def allclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> bool:
    xa = a.to_numpy() if isinstance(a, ndarray) else _np.asarray(a)
    xb = b.to_numpy() if isinstance(b, ndarray) else _np.asarray(b)
    return bool(_np.allclose(xa, xb, rtol=rtol, atol=atol, equal_nan=equal_nan))
