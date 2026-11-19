"""Comprehensive pytest suite verifying vknumpy drop-in parity with NumPy."""

import numpy as np
import pytest
import vknumpy as vnp


def test_device_info():
    info = vnp.device_info()
    assert isinstance(info["name"], str) and len(info["name"]) > 0
    assert info["backend"] in ("ggml-vulkan", "ggml-cpu")
    print("Active vknumpy device:", info)


# ----------------------------------------------------------------------
# Core creation, transfers, views, indexing
# ----------------------------------------------------------------------

def test_creation_and_transfers():
    a = vnp.zeros((3, 4))
    assert a.shape == (3, 4) and a.dtype == np.float32
    np.testing.assert_array_equal(a.to_numpy(), np.zeros((3, 4), dtype=np.float32))

    b = vnp.ones((2, 5), dtype=vnp.float16)
    assert b.dtype == np.float16
    np.testing.assert_array_equal(b.to_numpy(), np.ones((2, 5), dtype=np.float16))

    c = vnp.full((4, 3), 2.5)
    np.testing.assert_allclose(c.to_numpy(), np.full((4, 3), 2.5, dtype=np.float32))

    d = vnp.arange(0, 10, 2, dtype=vnp.float32)
    np.testing.assert_allclose(d.to_numpy(), np.arange(0, 10, 2, dtype=np.float32))

    e = vnp.linspace(-1.0, 1.0, 9)
    np.testing.assert_allclose(e.to_numpy(), np.linspace(-1.0, 1.0, 9, dtype=np.float32), atol=1e-6)

    eye = vnp.eye(4)
    np.testing.assert_allclose(eye.to_numpy(), np.eye(4, dtype=np.float32))


def test_reshape_transpose_slicing():
    x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = vnp.array(x_np)

    np.testing.assert_allclose(x.reshape(4, 6).to_numpy(), x_np.reshape(4, 6))
    np.testing.assert_allclose(x.reshape(-1, 8).to_numpy(), x_np.reshape(-1, 8))
    np.testing.assert_allclose(x.transpose(1, 2, 0).to_numpy(), x_np.transpose(1, 2, 0))
    np.testing.assert_allclose(x.swapaxes(-1, -2).to_numpy(), x_np.swapaxes(-1, -2))
    np.testing.assert_allclose(x[1].to_numpy(), x_np[1])
    np.testing.assert_allclose(x[0:1].to_numpy(), x_np[0:1])

    m_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    m = vnp.array(m_np)
    assert m.T._transposed_last2 is True
    np.testing.assert_allclose(m.T.to_numpy(), m_np.T)
    np.testing.assert_allclose(m.T.contiguous().to_numpy(), m_np.T)


def test_astype_gpu_cast():
    x_np = np.linspace(-3.0, 3.0, 64, dtype=np.float32)
    x = vnp.array(x_np)
    x16 = x.astype(vnp.float16)
    assert x16.dtype == np.float16
    np.testing.assert_allclose(x16.to_numpy(), x_np.astype(np.float16), atol=1e-3)
    x32 = x16.astype(vnp.float32)
    np.testing.assert_allclose(x32.to_numpy(), x_np.astype(np.float16).astype(np.float32), atol=1e-6)


# ----------------------------------------------------------------------
# GEMM & BLAS via llama.cpp ggml-vulkan
# ----------------------------------------------------------------------

@pytest.mark.parametrize("shape_a,shape_b", [
    ((16, 32), (32, 24)),
    ((128, 256), (256, 64)),
    ((1, 64), (64, 32)),
    ((64, 32), (32, 1)),
    ((4, 16, 32), (4, 32, 20)),
    ((4, 16, 32), (32, 20)),      # broadcast B
    ((16, 32), (4, 32, 20)),      # broadcast A
    ((2, 3, 8, 16), (2, 3, 16, 12)),
])
def test_matmul_f32(shape_a, shape_b):
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal(shape_a).astype(np.float32)
    b_np = rng.standard_normal(shape_b).astype(np.float32)

    out = vnp.array(a_np) @ vnp.array(b_np)
    ref = a_np @ b_np
    assert out.shape == ref.shape
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-4, atol=1e-3)


def test_matmul_transposed_fast_path():
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((32, 64)).astype(np.float32)
    w_np = rng.standard_normal((48, 64)).astype(np.float32)

    x = vnp.array(x_np)
    w = vnp.array(w_np)
    wt = w.T
    assert wt._transposed_last2 is True

    out = x @ wt
    ref = x_np @ w_np.T
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-4, atol=1e-3)


def test_matmul_f16():
    rng = np.random.default_rng(99)
    a_np = rng.standard_normal((64, 128)).astype(np.float16)
    b_np = rng.standard_normal((128, 64)).astype(np.float16)

    out = vnp.array(a_np) @ vnp.array(b_np)
    ref = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.float16)
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-2, atol=5e-2)


def test_dot_vdot_outer_inner_tensordot():
    rng = np.random.default_rng(12)
    u_np = rng.standard_normal(64).astype(np.float32)
    v_np = rng.standard_normal(64).astype(np.float32)
    w_np = rng.standard_normal(40).astype(np.float32)

    u, v, w = vnp.array(u_np), vnp.array(v_np), vnp.array(w_np)

    np.testing.assert_allclose(vnp.dot(u, v).to_numpy(), np.dot(u_np, v_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.vdot(u, v).to_numpy(), np.vdot(u_np, v_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.outer(u, w).to_numpy(), np.outer(u_np, w_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.inner(u, v).to_numpy(), np.inner(u_np, v_np), rtol=1e-4, atol=1e-4)

    # 1D @ 2D and 2D @ 1D
    m_np = rng.standard_normal((64, 40)).astype(np.float32)
    m = vnp.array(m_np)
    np.testing.assert_allclose((u @ m).to_numpy(), u_np @ m_np, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose((m @ w).to_numpy(), m_np @ w_np, rtol=1e-4, atol=1e-3)

    # tensordot
    t1_np = rng.standard_normal((3, 4, 5)).astype(np.float32)
    t2_np = rng.standard_normal((4, 5, 6)).astype(np.float32)
    np.testing.assert_allclose(
        vnp.tensordot(vnp.array(t1_np), vnp.array(t2_np), axes=2).to_numpy(),
        np.tensordot(t1_np, t2_np, axes=2),
        rtol=1e-4,
        atol=1e-3,
    )


def test_einsum_gpu_contractions():
    rng = np.random.default_rng(123)
    a_np = rng.standard_normal((16, 32)).astype(np.float32)
    b_np = rng.standard_normal((32, 24)).astype(np.float32)
    c_np = rng.standard_normal((24, 32)).astype(np.float32)

    a, b, c = vnp.array(a_np), vnp.array(b_np), vnp.array(c_np)
    np.testing.assert_allclose(vnp.einsum("ij,jk->ik", a, b).to_numpy(), np.einsum("ij,jk->ik", a_np, b_np), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(vnp.einsum("ij,kj->ik", a, c).to_numpy(), np.einsum("ij,kj->ik", a_np, c_np), rtol=1e-4, atol=1e-3)

    # Multi-head attention scores: bhqd,bhkd->bhqk
    q_np = rng.standard_normal((2, 4, 12, 16)).astype(np.float32)
    k_np = rng.standard_normal((2, 4, 12, 16)).astype(np.float32)
    np.testing.assert_allclose(
        vnp.einsum("bhqd,bhkd->bhqk", vnp.array(q_np), vnp.array(k_np)).to_numpy(),
        np.einsum("bhqd,bhkd->bhqk", q_np, k_np),
        rtol=1e-4,
        atol=1e-3,
    )


# ----------------------------------------------------------------------
# Elementwise ufuncs & broadcasting
# ----------------------------------------------------------------------

def test_binary_broadcasting_and_scalars():
    rng = np.random.default_rng(5)
    a_np = rng.standard_normal((8, 16)).astype(np.float32)
    b_np = rng.standard_normal((1, 16)).astype(np.float32)
    c_np = rng.standard_normal((8, 1)).astype(np.float32)

    a, b, c = vnp.array(a_np), vnp.array(b_np), vnp.array(c_np)

    np.testing.assert_allclose((a + b).to_numpy(), a_np + b_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose((b + c).to_numpy(), b_np + c_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose((a - c).to_numpy(), a_np - c_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose((a * b).to_numpy(), a_np * b_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose((a / (vnp.abs(b) + 0.5)).to_numpy(), a_np / (np.abs(b_np) + 0.5), rtol=1e-5, atol=1e-5)

    # Scalar fast paths
    np.testing.assert_allclose((a * 2.5 + 1.0).to_numpy(), a_np * 2.5 + 1.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose((3.0 - a).to_numpy(), 3.0 - a_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose((a / 4.0).to_numpy(), a_np / 4.0, rtol=1e-5, atol=1e-5)


def test_unary_ufuncs():
    x_np = np.linspace(0.2, 3.0, 128, dtype=np.float32)
    x = vnp.array(x_np)

    np.testing.assert_allclose((-x).to_numpy(), -x_np, atol=1e-6)
    np.testing.assert_allclose(vnp.abs(-x).to_numpy(), np.abs(-x_np), atol=1e-6)
    np.testing.assert_allclose(vnp.square(x).to_numpy(), np.square(x_np), rtol=1e-5)
    np.testing.assert_allclose(vnp.sqrt(x).to_numpy(), np.sqrt(x_np), rtol=1e-5)
    np.testing.assert_allclose(vnp.exp(x).to_numpy(), np.exp(x_np), rtol=1e-5)
    np.testing.assert_allclose(vnp.log(x).to_numpy(), np.log(x_np), rtol=1e-5)
    np.testing.assert_allclose(vnp.sin(x).to_numpy(), np.sin(x_np), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vnp.cos(x).to_numpy(), np.cos(x_np), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vnp.tanh(x).to_numpy(), np.tanh(x_np), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vnp.clip(x, 0.5, 2.0).to_numpy(), np.clip(x_np, 0.5, 2.0), atol=1e-6)


# ----------------------------------------------------------------------
# Reductions & linalg
# ----------------------------------------------------------------------

def test_reductions():
    rng = np.random.default_rng(77)
    x_np = rng.standard_normal((16, 32)).astype(np.float32)
    x = vnp.array(x_np)

    np.testing.assert_allclose(x.sum().to_numpy(), x_np.sum(), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(x.sum(axis=-1).to_numpy(), x_np.sum(axis=-1), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(x.sum(axis=0).to_numpy(), x_np.sum(axis=0), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(x.mean().to_numpy(), x_np.mean(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(x.mean(axis=-1, keepdims=True).to_numpy(), x_np.mean(axis=-1, keepdims=True), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(x.std(axis=-1).to_numpy(), x_np.std(axis=-1), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(x.argmax(axis=-1).to_numpy(), x_np.argmax(axis=-1))


def test_linalg():
    rng = np.random.default_rng(31)
    a_np = rng.standard_normal((16, 16)).astype(np.float32) + np.eye(16, dtype=np.float32) * 5.0
    b_np = rng.standard_normal((16, 4)).astype(np.float32)

    a, b = vnp.array(a_np), vnp.array(b_np)

    np.testing.assert_allclose(vnp.linalg.norm(a).to_numpy(), np.linalg.norm(a_np), rtol=1e-4)
    np.testing.assert_allclose(vnp.linalg.norm(a, axis=-1).to_numpy(), np.linalg.norm(a_np, axis=-1), rtol=1e-4)
    np.testing.assert_allclose(vnp.linalg.solve(a, b).to_numpy(), np.linalg.solve(a_np, b_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.linalg.matrix_power(a, 3).to_numpy(), np.linalg.matrix_power(a_np, 3), rtol=1e-4, atol=1e-2)

    m1 = vnp.array(rng.standard_normal((8, 32)).astype(np.float32))
    m2 = vnp.array(rng.standard_normal((32, 4)).astype(np.float32))
    m3 = vnp.array(rng.standard_normal((4, 16)).astype(np.float32))
    np.testing.assert_allclose(
        vnp.linalg.multi_dot([m1, m2, m3]).to_numpy(),
        np.linalg.multi_dot([m1.to_numpy(), m2.to_numpy(), m3.to_numpy()]),
        rtol=1e-4,
        atol=1e-3,
    )


# ----------------------------------------------------------------------
# Phase 1: Native ggml-vulkan creation, unary, manipulation, sort & solve_tri
# ----------------------------------------------------------------------

def test_phase1_gpu_creation_and_tri():
    # GPU diag (1D -> 2D)
    v_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(vnp.diag(vnp.array(v_np)).to_numpy(), np.diag(v_np))

    # GPU tril / triu
    m_np = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
    m = vnp.array(m_np)
    np.testing.assert_allclose(vnp.tril(m).to_numpy(), np.tril(m_np))
    np.testing.assert_allclose(vnp.tril(m, k=-1).to_numpy(), np.tril(m_np, k=-1))
    np.testing.assert_allclose(vnp.triu(m).to_numpy(), np.triu(m_np))
    np.testing.assert_allclose(vnp.triu(m, k=1).to_numpy(), np.triu(m_np, k=1))


def test_phase1_unary_math():
    x_np = np.array([-2.7, -1.5, -0.3, 0.0, 0.4, 1.5, 2.8], dtype=np.float32)
    x = vnp.array(x_np)

    np.testing.assert_allclose(vnp.floor(x).to_numpy(), np.floor(x_np))
    np.testing.assert_allclose(vnp.ceil(x).to_numpy(), np.ceil(x_np))
    np.testing.assert_allclose(vnp.trunc(x).to_numpy(), np.trunc(x_np))
    np.testing.assert_allclose(vnp.sign(x).to_numpy(), np.sign(x_np))
    np.testing.assert_allclose(vnp.expm1(x).to_numpy(), np.expm1(x_np), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vnp.sinh(x).to_numpy(), np.sinh(x_np), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vnp.cosh(x).to_numpy(), np.cosh(x_np), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vnp.cbrt(x).to_numpy(), np.cbrt(x_np), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(vnp.heaviside(x, 0.0).to_numpy(), np.heaviside(x_np, 0.0))

    pos_np = np.array([0.2, 1.0, 2.5, 10.0], dtype=np.float32)
    pos = vnp.array(pos_np)
    np.testing.assert_allclose(vnp.log1p(pos).to_numpy(), np.log1p(pos_np), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vnp.reciprocal(pos).to_numpy(), np.reciprocal(pos_np), rtol=1e-5)

    r_np = np.array([-2.3, -0.2, 0.3, 1.7, 2.8], dtype=np.float32)
    np.testing.assert_allclose(vnp.round(vnp.array(r_np)).to_numpy(), np.round(r_np))


def test_phase1_concat_pad_roll_tile_broadcast():
    a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    b_np = (np.arange(6, dtype=np.float32) + 10.0).reshape(2, 3)
    a, b = vnp.array(a_np), vnp.array(b_np)

    np.testing.assert_allclose(vnp.concatenate([a, b], axis=0).to_numpy(), np.concatenate([a_np, b_np], axis=0))
    np.testing.assert_allclose(vnp.concatenate([a, b], axis=1).to_numpy(), np.concatenate([a_np, b_np], axis=1))
    np.testing.assert_allclose(vnp.stack([a, b], axis=0).to_numpy(), np.stack([a_np, b_np], axis=0))
    np.testing.assert_allclose(vnp.vstack([a, b]).to_numpy(), np.vstack([a_np, b_np]))
    np.testing.assert_allclose(vnp.hstack([a, b]).to_numpy(), np.hstack([a_np, b_np]))
    np.testing.assert_allclose(vnp.dstack([a, b]).to_numpy(), np.dstack([a_np, b_np]))

    # GPU zero pad
    np.testing.assert_allclose(vnp.pad(a, ((1, 2), (2, 1))).to_numpy(), np.pad(a_np, ((1, 2), (2, 1))))

    # GPU roll
    np.testing.assert_allclose(vnp.roll(a, 2).to_numpy(), np.roll(a_np, 2))
    np.testing.assert_allclose(vnp.roll(a, 1, axis=0).to_numpy(), np.roll(a_np, 1, axis=0))
    np.testing.assert_allclose(vnp.roll(a, -1, axis=1).to_numpy(), np.roll(a_np, -1, axis=1))

    # GPU tile & broadcast_to
    np.testing.assert_allclose(vnp.tile(a, (2, 3)).to_numpy(), np.tile(a_np, (2, 3)))
    row_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(vnp.broadcast_to(vnp.array(row_np), (4, 3)).to_numpy(), np.broadcast_to(row_np, (4, 3)))


def test_phase1_take_cumsum_argsort_sort_topk():
    rng = np.random.default_rng(2026)
    x_np = rng.standard_normal((4, 16)).astype(np.float32)
    x = vnp.array(x_np)

    # GPU take
    idx_np = np.array([2, 0, 3, 1], dtype=np.int32)
    np.testing.assert_allclose(vnp.take(x, idx_np, axis=0).to_numpy(), np.take(x_np, idx_np, axis=0))
    np.testing.assert_allclose(vnp.take(x, np.array([0, 5, 17, 63], dtype=np.int32)).to_numpy(), np.take(x_np, [0, 5, 17, 63]))

    # GPU cumsum
    np.testing.assert_allclose(vnp.cumsum(x, axis=-1).to_numpy(), np.cumsum(x_np, axis=-1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.cumsum(x, axis=0).to_numpy(), np.cumsum(x_np, axis=0), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.cumsum(x).to_numpy(), np.cumsum(x_np), rtol=1e-4, atol=1e-4)

    # GPU argsort & sort
    np.testing.assert_array_equal(vnp.argsort(x, axis=-1).to_numpy(), np.argsort(x_np, axis=-1))
    np.testing.assert_allclose(vnp.sort(x[0]).to_numpy(), np.sort(x_np[0]))

    # GPU top_k (check returned indices match the top-k values)
    k = 4
    tk_idx = vnp.top_k(x, k, axis=-1).to_numpy()
    assert tk_idx.shape == (4, k)
    for row in range(4):
        got_vals = np.sort(x_np[row, tk_idx[row]])
        ref_vals = np.sort(x_np[row])[-k:]
        np.testing.assert_allclose(got_vals, ref_vals)


def test_phase1_solve_triangular():
    rng = np.random.default_rng(55)
    raw = rng.standard_normal((8, 8)).astype(np.float32)
    L_np = np.tril(raw)
    np.fill_diagonal(L_np, np.abs(np.diag(L_np)) + 2.0)
    B_np = rng.standard_normal((8, 3)).astype(np.float32)

    X_vk = vnp.linalg.solve_triangular(vnp.array(L_np), vnp.array(B_np), lower=True)
    X_ref = np.linalg.solve(L_np, B_np)
    np.testing.assert_allclose(X_vk.to_numpy(), X_ref, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------
# Phase 2: ggml-vulkan comparisons/where/logicals + ncnn Vulkan kernels
# ----------------------------------------------------------------------

def test_phase2_comparisons_where_and_logicals():
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_np = np.array([[2.0, 2.0, 1.0], [5.0, 5.0, 2.0]], dtype=np.float32)
    a, b = vnp.array(a_np), vnp.array(b_np)

    np.testing.assert_array_equal((a > b).to_numpy(), a_np > b_np)
    np.testing.assert_array_equal((a < b).to_numpy(), a_np < b_np)
    np.testing.assert_array_equal((a >= b).to_numpy(), a_np >= b_np)
    np.testing.assert_array_equal((a <= b).to_numpy(), a_np <= b_np)
    np.testing.assert_array_equal((a == b).to_numpy(), a_np == b_np)
    np.testing.assert_array_equal((a != b).to_numpy(), a_np != b_np)

    np.testing.assert_allclose(vnp.maximum(a, b).to_numpy(), np.maximum(a_np, b_np), atol=1e-5)
    np.testing.assert_allclose(vnp.minimum(a, b).to_numpy(), np.minimum(a_np, b_np), atol=1e-5)
    np.testing.assert_allclose(vnp.where(a > b, a, b).to_numpy(), np.where(a_np > b_np, a_np, b_np))

    m1 = a > 2.0
    m2 = b < 5.0
    np.testing.assert_array_equal(vnp.logical_and(m1, m2).to_numpy(), np.logical_and(a_np > 2.0, b_np < 5.0))
    np.testing.assert_array_equal(vnp.logical_or(m1, m2).to_numpy(), np.logical_or(a_np > 2.0, b_np < 5.0))
    np.testing.assert_array_equal(vnp.logical_xor(m1, m2).to_numpy(), np.logical_xor(a_np > 2.0, b_np < 5.0))
    np.testing.assert_array_equal(vnp.logical_not(m1).to_numpy(), np.logical_not(a_np > 2.0))

    assert bool(vnp.any(a > 5.0).item()) is True
    assert bool(vnp.all(a > 0.0).item()) is True
    assert bool(vnp.all(a > 2.0).item()) is False
    assert int(vnp.count_nonzero(vnp.array([0.0, 1.0, 0.0, 3.5])).item()) == 2
    assert vnp.array_equal(a, vnp.array(a_np)) is True
    assert vnp.array_equal(a, b) is False
    np.testing.assert_array_equal(vnp.isclose(a, b, atol=1.0).to_numpy(), np.isclose(a_np, b_np, atol=1.0))


def test_phase2_ncnn_math_and_reductions():
    from vknumpy._backend import lib
    assert lib().vkn_ncnn_available() == 1

    # Inverse trig & hyperbolic (ncnn Vulkan UnaryOp)
    x_np = np.array([-0.8, -0.3, 0.0, 0.4, 0.9], dtype=np.float32)
    x = vnp.array(x_np)
    np.testing.assert_allclose(vnp.arcsin(x).to_numpy(), np.arcsin(x_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.arccos(x).to_numpy(), np.arccos(x_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.arctan(x).to_numpy(), np.arctan(x_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.arcsinh(x).to_numpy(), np.arcsinh(x_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.arctanh(x).to_numpy(), np.arctanh(x_np), rtol=1e-4, atol=1e-4)
    c_np = np.array([1.2, 2.0, 3.5], dtype=np.float32)
    np.testing.assert_allclose(vnp.arccosh(vnp.array(c_np)).to_numpy(), np.arccosh(c_np), rtol=1e-4, atol=1e-4)

    # Binary math (ncnn Vulkan BinaryOp: pow, atan2, fmod, remainder, floor_divide, logaddexp)
    a_np = np.array([1.5, 2.0, 3.5, 4.0], dtype=np.float32)
    b_np = np.array([2.0, 3.0, 1.5, 0.5], dtype=np.float32)
    a, b = vnp.array(a_np), vnp.array(b_np)
    np.testing.assert_allclose(vnp.power(a, b).to_numpy(), np.power(a_np, b_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.arctan2(a, b).to_numpy(), np.arctan2(a_np, b_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.fmod(a, b).to_numpy(), np.fmod(a_np, b_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.remainder(a, b).to_numpy(), np.remainder(a_np, b_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.floor_divide(a, b).to_numpy(), np.floor_divide(a_np, b_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vnp.logaddexp(a, b).to_numpy(), np.logaddexp(a_np, b_np), rtol=1e-4, atol=1e-4)

    # Reductions (ncnn Vulkan Reduction: max, min, prod)
    m_np = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 3.0]], dtype=np.float32)
    m = vnp.array(m_np)
    np.testing.assert_allclose(vnp.max(m).to_numpy(), np.max(m_np))
    np.testing.assert_allclose(vnp.min(m).to_numpy(), np.min(m_np))
    np.testing.assert_allclose(vnp.prod(m).to_numpy(), np.prod(m_np), rtol=1e-4)
    np.testing.assert_allclose(vnp.max(m, axis=-1).to_numpy(), np.max(m_np, axis=-1))
    np.testing.assert_allclose(vnp.min(m, axis=0).to_numpy(), np.min(m_np, axis=0))
    np.testing.assert_allclose(vnp.prod(m, axis=-1).to_numpy(), np.prod(m_np, axis=-1), rtol=1e-4)


