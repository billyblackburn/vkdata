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
