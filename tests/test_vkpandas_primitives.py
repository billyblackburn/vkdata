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


def test_track_c_diff_unique_bincount():
    # diff (1D and 2D along axes)
    a_np = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    np.testing.assert_allclose(vnp.diff(vnp.array(a_np)).to_numpy(), np.diff(a_np))
    np.testing.assert_allclose(vnp.diff(vnp.array(a_np), n=2).to_numpy(), np.diff(a_np, n=2))

    m_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(vnp.diff(vnp.array(m_np), axis=0).to_numpy(), np.diff(m_np, axis=0))
    np.testing.assert_allclose(vnp.diff(vnp.array(m_np), axis=1).to_numpy(), np.diff(m_np, axis=1))

    # unique with return_inverse and return_counts
    x_np = np.array([5, 2, 5, 1, 2, 2, 9, 1], dtype=np.int32)
    u, inv, cnt = vnp.unique(vnp.array(x_np), return_inverse=True, return_counts=True)
    u_ref, inv_ref, cnt_ref = np.unique(x_np, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(u.to_numpy(), u_ref)
    np.testing.assert_array_equal(inv.to_numpy(), inv_ref)
    np.testing.assert_array_equal(cnt.to_numpy(), cnt_ref)
    # Reconstruct original via u[inv]
    np.testing.assert_array_equal(u[inv].to_numpy(), x_np)

    # bincount (unweighted and weighted)
    bc_np = np.array([0, 2, 2, 5, 2, 0], dtype=np.int32)
    w_np = np.array([1.5, 2.0, 3.0, 4.5, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(vnp.bincount(vnp.array(bc_np)).to_numpy(), np.bincount(bc_np))
    np.testing.assert_array_equal(vnp.bincount(vnp.array(bc_np), minlength=8).to_numpy(), np.bincount(bc_np, minlength=8))
    np.testing.assert_allclose(vnp.bincount(vnp.array(bc_np), weights=vnp.array(w_np)).to_numpy(), np.bincount(bc_np, weights=w_np), rtol=1e-5)


def test_track_c_groupby_agg():
    rng = np.random.default_rng(7)
    n = 512
    keys_np = rng.integers(0, 12, size=n, dtype=np.int32)
    vals_np = rng.uniform(1.0, 50.0, size=n).astype(np.float32)

    keys = vnp.array(keys_np)
    vals = vnp.array(vals_np)

    for op in ("count", "sum", "mean", "var", "std"):
        uk, res = vnp.groupby_agg(keys, vals, op=op)
        uk_np = uk.to_numpy()
        res_np = res.to_numpy()

        ref_keys = np.unique(keys_np)
        np.testing.assert_array_equal(uk_np, ref_keys)
        for i, k in enumerate(ref_keys):
            grp = vals_np[keys_np == k]
            if op == "count":
                assert res_np[i] == len(grp)
            elif op == "sum":
                np.testing.assert_allclose(res_np[i], grp.sum(), rtol=1e-4)
            elif op == "mean":
                np.testing.assert_allclose(res_np[i], grp.mean(), rtol=1e-4)
            elif op == "var":
                np.testing.assert_allclose(res_np[i], grp.var(), rtol=1e-3, atol=1e-2)
            elif op == "std":
                np.testing.assert_allclose(res_np[i], grp.std(), rtol=1e-3, atol=1e-2)


def test_track_d_searchsorted_and_isin():
    a_np = np.array([10.0, 20.0, 20.0, 30.0, 40.0], dtype=np.float32)
    v_np = np.array([5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 45.0], dtype=np.float32)
    a, v = vnp.array(a_np), vnp.array(v_np)

    np.testing.assert_array_equal(vnp.searchsorted(a, v, side="left").to_numpy(), np.searchsorted(a_np, v_np, side="left"))
    np.testing.assert_array_equal(vnp.searchsorted(a, v, side="right").to_numpy(), np.searchsorted(a_np, v_np, side="right"))

    # 2D query shape + int32 arrays
    ai_np = np.array([1, 3, 3, 7, 9], dtype=np.int32)
    vi_np = np.array([[0, 3, 5], [7, 9, 12]], dtype=np.int32)
    np.testing.assert_array_equal(
        vnp.searchsorted(vnp.array(ai_np), vnp.array(vi_np), side="left").to_numpy(),
        np.searchsorted(ai_np, vi_np, side="left"),
    )
    np.testing.assert_array_equal(
        vnp.searchsorted(vnp.array(ai_np), vnp.array(vi_np), side="right").to_numpy(),
        np.searchsorted(ai_np, vi_np, side="right"),
    )

    # isin & invert=True
    el_np = np.array([1, 2, 3, 4, 5, 7, 9, 10], dtype=np.int32)
    test_np = np.array([9, 3, 1, 7], dtype=np.int32)
    np.testing.assert_array_equal(vnp.isin(vnp.array(el_np), vnp.array(test_np)).to_numpy(), np.isin(el_np, test_np))
    np.testing.assert_array_equal(
        vnp.isin(vnp.array(el_np), vnp.array(test_np), invert=True).to_numpy(),
        np.isin(el_np, test_np, invert=True),
    )


def test_track_d_gpu_dataframe_inner_join():
    # Orders table (left) joined with Users table (right, unique user_id)
    left_uid_np = np.array([103, 101, 999, 102, 101, 104, 103], dtype=np.int32)
    left_amt_np = np.array([15.0, 20.0, 5.0, 30.0, 10.0, 40.0, 25.0], dtype=np.float32)

    right_uid_np = np.array([104, 102, 101, 103], dtype=np.int32)
    right_score_np = np.array([4.0, 2.0, 1.0, 3.0], dtype=np.float32)

    l_idx, r_idx = vnp.merge_indices(vnp.array(left_uid_np), vnp.array(right_uid_np))
    joined_uid = vnp.take(vnp.array(left_uid_np), l_idx).to_numpy()
    joined_amt = vnp.take(vnp.array(left_amt_np), l_idx).to_numpy()
    joined_score = vnp.take(vnp.array(right_score_np), r_idx).to_numpy()

    # Row 2 (uid=999) has no match and must be dropped; remaining 6 rows match exact right_score
    np.testing.assert_array_equal(joined_uid, [103, 101, 102, 101, 104, 103])
    np.testing.assert_allclose(joined_amt, [15.0, 20.0, 30.0, 10.0, 40.0, 25.0])
    np.testing.assert_allclose(joined_score, [3.0, 1.0, 2.0, 1.0, 4.0, 3.0])


def test_track_e_nan_predicates_and_reductions():
    x_np = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -3.5], dtype=np.float32)
    x = vnp.array(x_np)

    np.testing.assert_array_equal(vnp.isnan(x).to_numpy(), np.isnan(x_np))
    np.testing.assert_array_equal(vnp.isinf(x).to_numpy(), np.isinf(x_np))
    np.testing.assert_array_equal(vnp.isfinite(x).to_numpy(), np.isfinite(x_np))
    np.testing.assert_allclose(
        vnp.nan_to_num(x, nan=-1.0, posinf=99.0, neginf=-99.0).to_numpy(),
        np.nan_to_num(x_np, nan=-1.0, posinf=99.0, neginf=-99.0),
    )

    m_np = np.array([[1.0, np.nan, 3.0, 4.0], [np.nan, 2.0, 6.0, np.nan]], dtype=np.float32)
    m = vnp.array(m_np)

    np.testing.assert_allclose(vnp.nansum(m).to_numpy(), np.nansum(m_np), rtol=1e-5)
    np.testing.assert_allclose(vnp.nansum(m, axis=-1).to_numpy(), np.nansum(m_np, axis=-1), rtol=1e-5)
    np.testing.assert_allclose(vnp.nanmean(m).to_numpy(), np.nanmean(m_np), rtol=1e-5)
    np.testing.assert_allclose(vnp.nanmean(m, axis=0).to_numpy(), np.nanmean(m_np, axis=0), rtol=1e-5)
    np.testing.assert_allclose(vnp.nanvar(m, axis=-1).to_numpy(), np.nanvar(m_np, axis=-1), rtol=1e-4)
    np.testing.assert_allclose(vnp.nanstd(m, axis=-1).to_numpy(), np.nanstd(m_np, axis=-1), rtol=1e-4)
    np.testing.assert_allclose(vnp.nanmax(m).to_numpy(), np.nanmax(m_np))
    np.testing.assert_allclose(vnp.nanmax(m, axis=-1).to_numpy(), np.nanmax(m_np, axis=-1))
    np.testing.assert_allclose(vnp.nanmin(m).to_numpy(), np.nanmin(m_np))
    np.testing.assert_allclose(vnp.nanmin(m, axis=0).to_numpy(), np.nanmin(m_np, axis=0))

    # Forward-fill & backward-fill
    s_np = np.array([10.0, np.nan, np.nan, 20.0, np.nan, 30.0], dtype=np.float32)
    np.testing.assert_allclose(vnp.ffill(vnp.array(s_np)).to_numpy(), [10.0, 10.0, 10.0, 20.0, 20.0, 30.0])
    s2_np = np.array([np.nan, 10.0, np.nan, np.nan, 20.0], dtype=np.float32)
    np.testing.assert_allclose(vnp.bfill(vnp.array(s2_np)).to_numpy(), [10.0, 10.0, 20.0, 20.0, 20.0])


def test_track_f_rolling_lexsort_and_string_factorize():
    # GPU rolling_sum & rolling_mean (matching pandas leading NaNs)
    x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    rs = vnp.rolling_sum(vnp.array(x_np), window=3).to_numpy()
    rm = vnp.rolling_mean(vnp.array(x_np), window=3).to_numpy()
    assert np.isnan(rs[0]) and np.isnan(rs[1])
    np.testing.assert_allclose(rs[2:], [6.0, 9.0, 12.0, 15.0], rtol=1e-5)
    assert np.isnan(rm[0]) and np.isnan(rm[1])
    np.testing.assert_allclose(rm[2:], [2.0, 3.0, 4.0, 5.0], rtol=1e-5)

    # Multi-column lexsort (primary=dept, secondary=salary)
    dept_np = np.array([2, 1, 2, 1, 1], dtype=np.int32)
    sal_np = np.array([50.0, 90.0, 30.0, 60.0, 70.0], dtype=np.float32)
    order = vnp.lexsort((vnp.array(sal_np), vnp.array(dept_np)))
    ref_order = np.lexsort((sal_np, dept_np))
    np.testing.assert_array_equal(order.to_numpy(), ref_order)

    # Dictionary-encoded string column + GPU groupby_agg!
    cities = ["nyc", "la", "nyc", "chi", "la", "nyc"]
    sales = vnp.array([10.0, 20.0, 30.0, 5.0, 15.0, 50.0], dtype=vnp.float32)
    codes, uniques = vnp.factorize(cities)
    assert codes.dtype == np.int32
    assert list(uniques) == ["nyc", "la", "chi"]

    u_codes, totals = vnp.groupby_agg(codes, sales, op="sum")
    result_map = {str(uniques[int(c)]): float(t) for c, t in zip(u_codes.to_numpy(), totals.to_numpy())}
    assert result_map == {"nyc": 90.0, "la": 35.0, "chi": 5.0}




