"""Tests for vkpandas readiness primitives (Track A & Track B)."""

import numpy as np
import vknumpy as vnp


def test_track_a_int32_gpu_ops():
    k_np = np.array([30, 10, 40, 10, 20, 50], dtype=np.int32)
    k = vnp.array(k_np)
    assert k.dtype == np.int32

    # GPU int32 <-> float32 cast
    kf = k.astype(vnp.float32)
    assert kf.dtype == np.float32
    np.testing.assert_allclose(kf.to_numpy(), k_np.astype(np.float32))
    np.testing.assert_array_equal(kf.astype(vnp.int32).to_numpy(), k_np)

    # GPU int32 comparisons
    np.testing.assert_array_equal((k > 20).to_numpy(), k_np > 20)
    np.testing.assert_array_equal((k == 10).to_numpy(), k_np == 10)
    np.testing.assert_array_equal((k <= 30).to_numpy(), k_np <= 30)

    # GPU int32 take (native get_rows_i32)
    idx = vnp.array(np.array([5, 1, 0, 3], dtype=np.int32))
    got = vnp.take(k, idx)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got.to_numpy(), np.take(k_np, [5, 1, 0, 3]))

    # GPU int32 argsort & cumsum
    order = vnp.argsort(k)
    np.testing.assert_array_equal(vnp.take(k, order).to_numpy(), np.sort(k_np))
    np.testing.assert_array_equal(vnp.cumsum(k).to_numpy(), np.cumsum(k_np))


def test_track_b_scatter_and_put():
    # 2D row scatter via ggml_set_rows
    dst_np = np.zeros((5, 3), dtype=np.float32)
    src_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    idx_np = np.array([3, 1], dtype=np.int32)

    out = vnp.scatter_rows(vnp.array(dst_np), idx_np, vnp.array(src_np))
    ref = dst_np.copy()
    ref[idx_np] = src_np
    np.testing.assert_allclose(out.to_numpy(), ref)

    # 1D in-place put
    a_np = np.arange(8, dtype=np.float32)
    a = vnp.array(a_np)
    vnp.put(a, np.array([1, 4, 6], dtype=np.int32), np.array([99.0, 88.0, 77.0], dtype=np.float32))
    np.put(a_np, [1, 4, 6], [99.0, 88.0, 77.0])
    np.testing.assert_allclose(a.to_numpy(), a_np)


def test_track_b_stream_compaction_nonzero_and_compress():
    # Standard mixed mask
    x_np = np.array([0.0, 3.5, 0.0, -1.2, 4.0, 0.0, 2.0], dtype=np.float32)
    x = vnp.array(x_np)
    np.testing.assert_array_equal(vnp.flatnonzero(x).to_numpy(), np.flatnonzero(x_np))
    np.testing.assert_array_equal(vnp.nonzero(x)[0].to_numpy(), np.nonzero(x_np)[0])

    # All-False and All-True edge cases
    z = vnp.zeros((16,), dtype=vnp.float32)
    assert vnp.flatnonzero(z).shape == (0,)
    o = vnp.ones((16,), dtype=vnp.float32)
    np.testing.assert_array_equal(vnp.flatnonzero(o).to_numpy(), np.arange(16, dtype=np.int32))

    # 2D nonzero
    m_np = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32)
    r_vk, c_vk = vnp.nonzero(vnp.array(m_np))
    r_ref, c_ref = np.nonzero(m_np)
    np.testing.assert_array_equal(r_vk.to_numpy(), r_ref)
    np.testing.assert_array_equal(c_vk.to_numpy(), c_ref)

    # compress along axis=0 (DataFrame row filtering)
    table_np = np.arange(18, dtype=np.float32).reshape(6, 3)
    cond_np = np.array([True, False, True, False, False, True], dtype=np.bool_)
    out = vnp.compress(vnp.array(cond_np), vnp.array(table_np), axis=0)
    np.testing.assert_allclose(out.to_numpy(), np.compress(cond_np, table_np, axis=0))


def test_track_b_boolean_mask_and_fancy_indexing_dataframe_workflow():
    rng = np.random.default_rng(42)
    n = 1024
    ids_np = np.arange(n, dtype=np.int32)
    price_np = rng.uniform(1.0, 100.0, size=n).astype(np.float32)
    qty_np = rng.integers(1, 20, size=n, dtype=np.int32)

    ids = vnp.array(ids_np)
    price = vnp.array(price_np)
    qty = vnp.array(qty_np)

    # Filter: df[df["price"] > 50.0]
    mask = price > 50.0
    f_ids = ids[mask]
    f_price = price[mask]
    f_qty = qty[mask]

    mask_np = price_np > 50.0
    np.testing.assert_array_equal(f_ids.to_numpy(), ids_np[mask_np])
    np.testing.assert_allclose(f_price.to_numpy(), price_np[mask_np])
    np.testing.assert_array_equal(f_qty.to_numpy(), qty_np[mask_np])

    # Sort filtered DataFrame by price ascending using GPU argsort + fancy indexing
    order = vnp.argsort(f_price)
    s_price = f_price[order]
    s_ids = f_ids[order]

    order_np = np.argsort(price_np[mask_np])
    np.testing.assert_allclose(s_price.to_numpy(), price_np[mask_np][order_np])
    np.testing.assert_array_equal(s_ids.to_numpy(), ids_np[mask_np][order_np])
