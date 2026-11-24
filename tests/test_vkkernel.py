"""Unit tests for @vnp.kernel (Python AST -> GLSL 450 -> SPIR-V zero-copy Vulkan execution)."""

import numpy as np
import vknumpy as vnp
from vknumpy import barrier, global_id_x, local_id_x, shared_array


@vnp.kernel(local_size=(64, 1, 1))
def _k_map_add10(out, a, n: int):
    i = global_id_x()
    if i < n:
        out[i] = a[i] + 10.0


@vnp.kernel(local_size=(64, 1, 1))
def _k_shared_neighbor_avg(out, a, n: int):
    tile = shared_array(64)
    i = global_id_x()
    tx = local_id_x()
    tile[tx] = a[i] if i < n else 0.0
    barrier()
    if i < n:
        prev_v = tile[tx - 1] if tx > 0 else tile[tx]
        next_v = tile[tx + 1] if tx < 63 else tile[tx]
        out[i] = (prev_v + tile[tx] + next_v) / 3.0


def test_python_ast_vulkan_kernel_map():
    assert "#version 450" in _k_map_add10.glsl
    assert len(_k_map_add10.spirv_bytes) > 100

    n = 1000  # intentionally not a multiple of 64 to test bounds guard!
    a_np = np.arange(n, dtype=np.float32)
    a = vnp.array(a_np)
    out = vnp.zeros((n,), dtype=np.float32)

    _k_map_add10[(n + 63) // 64](out, a, n)
    np.testing.assert_allclose(out.to_numpy(), a_np + 10.0, atol=1e-5)


def test_python_ast_vulkan_kernel_shared_memory_barrier():
    assert "shared float tile[64];" in _k_shared_neighbor_avg.glsl
    assert "barrier();" in _k_shared_neighbor_avg.glsl

    n = 64
    a_np = np.arange(n, dtype=np.float32)
    a = vnp.array(a_np)
    out = vnp.zeros((n,), dtype=np.float32)

    _k_shared_neighbor_avg[1](out, a, n)
    res = out.to_numpy()
    # Interior points of linear sequence [0, 1, 2, ...] average to themselves
    np.testing.assert_allclose(res[1:-1], a_np[1:-1], atol=1e-5)
