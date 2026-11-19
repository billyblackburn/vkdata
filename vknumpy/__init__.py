"""vknumpy: NumPy drop-in accelerated by llama.cpp's ggml-vulkan GEMM & BLAS."""

from __future__ import annotations

import numpy as _np

# Dtypes & constants matching NumPy namespace
float16 = _np.float16
float32 = _np.float32
float64 = _np.float64
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
uint8 = _np.uint8
bool_ = _np.bool_
dtype = _np.dtype

pi = _np.pi
e = _np.e
inf = _np.inf
nan = _np.nan
newaxis = None

# Core ndarray & submodules
from .ndarray import ndarray
from . import linalg
from .device import device_info, device_name, is_vulkan_available, memory_info

# Creation & random
from .creation import (
    arange,
    array,
    asarray,
    diag,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    identity,
    linspace,
    ones,
    ones_like,
    random,
    tril,
    triu,
    zeros,
    zeros_like,
)

# GEMM, BLAS, ufuncs, reductions, manipulation
from .ops import (
    abs,
    absolute,
    add,
    all,
    allclose,
    any,
    argmax,
    argmin,
    argsort,
    array_split,
    broadcast_to,
    cbrt,
    ceil,
    clip,
    concatenate,
    copy,
    cos,
    cosh,
    cumsum,
    divide,
    dot,
    dstack,
    einsum,
    exp,
    exp2,
    expand_dims,
    expm1,
    flip,
    floor,
    gelu,
    heaviside,
    hsplit,
    hstack,
    inner,
    log,
    log10,
    log1p,
    log2,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    moveaxis,
    multiply,
    negative,
    outer,
    pad,
    power,
    prod,
    reciprocal,
    relu,
    reshape,
    rint,
    roll,
    round,
    sigmoid,
    sign,
    silu,
    sin,
    sinh,
    sort,
    split,
    sqrt,
    square,
    squeeze,
    stack,
    std,
    subtract,
    sum,
    swapaxes,
    take,
    tan,
    tanh,
    tensordot,
    tile,
    top_k,
    transpose,
    true_divide,
    trunc,
    var,
    vdot,
    vsplit,
    vstack,
    where,
)

__version__ = "0.1.0"
