"""Linear algebra routines (vknumpy.linalg) accelerated with ggml-vulkan GEMM/BLAS."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union
import numpy as _np

from ._backend import NATIVE_FLOAT_DTYPES
from .creation import asarray, eye
from .ndarray import ndarray
from .ops import abs as vk_abs, matmul, sqrt, square, sum as vk_sum


def norm(x: Any, ord: Any = None, axis: Any = None, keepdims: bool = False) -> ndarray:
    """Matrix or vector norm, using GPU reductions for L2 / Frobenius / L1 norms."""
    xa = asarray(x)
    if xa.dtype in NATIVE_FLOAT_DTYPES and xa.size > 0:
        # L2 / Frobenius norm (most common: ord is None, 'fro', or 2 on vectors/single axis)
        if ord is None or ord == "fro" or (ord == 2 and (xa.ndim == 1 or isinstance(axis, int))):
            if axis is None or isinstance(axis, int):
                return sqrt(vk_sum(square(xa), axis=axis, keepdims=keepdims))
        if ord == 1 and (xa.ndim == 1 or isinstance(axis, int)):
            return vk_sum(vk_abs(xa), axis=axis, keepdims=keepdims)

    res = _np.linalg.norm(xa.to_numpy(), ord=ord, axis=axis, keepdims=keepdims)
    return ndarray._from_numpy(_np.asarray(res))


def multi_dot(arrays: Sequence[Any]) -> ndarray:
    """Compute the dot product of two or more arrays using optimal parenthesization on GPU."""
    arrs = [asarray(a) for a in arrays]
    n = len(arrs)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    if n == 2:
        return matmul(arrs[0], arrs[1])

    # Dynamic programming for optimal matrix chain multiplication order
    dims = [arrs[0].shape[0]] + [a.shape[-1] for a in arrs]
    cost = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            best = float("inf")
            for k in range(i, j):
                c = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if c < best:
                    best = c
                    split[i][j] = k
            cost[i][j] = best

    def _eval(i: int, j: int) -> ndarray:
        if i == j:
            return arrs[i]
        k = split[i][j]
        return matmul(_eval(i, k), _eval(k + 1, j))

    return _eval(0, n - 1)


def matrix_power(a: Any, n: int) -> ndarray:
    """Raise a square matrix to integer power n using repeated squaring on GPU."""
    xa = asarray(a)
    if xa.ndim < 2 or xa.shape[-1] != xa.shape[-2]:
        raise ValueError("input must be a square matrix (or stack of square matrices)")
    if n == 0:
        return eye(xa.shape[-1], dtype=xa.dtype)
    if n < 0:
        xa = inv(xa)
        n = -n
    result: Optional[ndarray] = None
    base = xa
    while n > 0:
        if n & 1:
            result = base if result is None else matmul(result, base)
        n >>= 1
        if n > 0:
            base = matmul(base, base)
    assert result is not None
    return result


def solve(a: Any, b: Any) -> ndarray:
    xa = asarray(a)
    xb = asarray(b)
    return ndarray._from_numpy(_np.asarray(_np.linalg.solve(xa.to_numpy(), xb.to_numpy())))


def inv(a: Any) -> ndarray:
    return ndarray._from_numpy(_np.asarray(_np.linalg.inv(asarray(a).to_numpy())))


def pinv(a: Any, rcond: float = 1e-15) -> ndarray:
    return ndarray._from_numpy(_np.asarray(_np.linalg.pinv(asarray(a).to_numpy(), rcond=rcond)))


def det(a: Any) -> ndarray:
    return ndarray._from_numpy(_np.asarray(_np.linalg.det(asarray(a).to_numpy())))


def slogdet(a: Any) -> Tuple[ndarray, ndarray]:
    sign, logdet = _np.linalg.slogdet(asarray(a).to_numpy())
    return ndarray._from_numpy(_np.asarray(sign)), ndarray._from_numpy(_np.asarray(logdet))


def cholesky(a: Any) -> ndarray:
    return ndarray._from_numpy(_np.asarray(_np.linalg.cholesky(asarray(a).to_numpy())))


def qr(a: Any, mode: str = "reduced") -> Any:
    res = _np.linalg.qr(asarray(a).to_numpy(), mode=mode)
    if isinstance(res, tuple):
        return tuple(ndarray._from_numpy(_np.asarray(r)) for r in res)
    return ndarray._from_numpy(_np.asarray(res))


def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Any:
    res = _np.linalg.svd(asarray(a).to_numpy(), full_matrices=full_matrices, compute_uv=compute_uv)
    if isinstance(res, tuple):
        return tuple(ndarray._from_numpy(_np.asarray(r)) for r in res)
    return ndarray._from_numpy(_np.asarray(res))


def eig(a: Any) -> Tuple[ndarray, ndarray]:
    w, v = _np.linalg.eig(asarray(a).to_numpy())
    return ndarray._from_numpy(_np.asarray(w)), ndarray._from_numpy(_np.asarray(v))


def eigh(a: Any, UPLO: str = "L") -> Tuple[ndarray, ndarray]:
    w, v = _np.linalg.eigh(asarray(a).to_numpy(), UPLO=UPLO)
    return ndarray._from_numpy(_np.asarray(w)), ndarray._from_numpy(_np.asarray(v))


def matrix_rank(A: Any, tol: Any = None) -> int:
    return int(_np.linalg.matrix_rank(asarray(A).to_numpy(), tol=tol))


def trace(a: Any, offset: int = 0) -> ndarray:
    return ndarray._from_numpy(_np.asarray(_np.trace(asarray(a).to_numpy(), offset=offset)))
