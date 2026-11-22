"""Vulkan GPU K-NN graph construction: exact chunked GEMM brute-force & NNDescent."""

from __future__ import annotations

from typing import Tuple

import numpy as _np
import vknumpy as vnp


def _to_f32_2d(X) -> _np.ndarray:
    arr = X.to_numpy() if isinstance(X, vnp.ndarray) else _np.asarray(X)
    arr = _np.ascontiguousarray(arr, dtype=_np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (N, D), got shape {arr.shape}")
    return arr


def brute_force_knn(
    X,
    k: int = 15,
    metric: str = "euclidean",
    chunk_size: int = 2048,
) -> Tuple[_np.ndarray, _np.ndarray]:
    """Exact K-NN graph via chunked Vulkan GPU GEMM + ``vnp.top_k``.

    Excludes self-neighbors (distance 0 at index ``i``).

    Returns
    -------
    indices : ndarray of shape (N, k), int32
    distances : ndarray of shape (N, k), float32
    """
    X_np = _to_f32_2d(X)
    n, d = X_np.shape
    k_eff = min(int(k), max(1, n - 1))
    k_query = min(k_eff + 1, n)

    if metric == "cosine":
        norms = _np.linalg.norm(X_np, axis=1, keepdims=True)
        X_use = X_np / _np.maximum(norms, 1e-8)
    elif metric == "euclidean":
        X_use = X_np
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'euclidean' or 'cosine'.")

    X_gpu = vnp.array(X_use, dtype=_np.float32)
    X_T = X_gpu.T.contiguous()
    sq_all = vnp.sum(X_gpu * X_gpu, axis=1).reshape(1, n) if metric == "euclidean" else None

    out_idx = _np.empty((n, k_eff), dtype=_np.int32)
    out_dist = _np.empty((n, k_eff), dtype=_np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        bs = end - start
        Xc = vnp.array(X_use[start:end], dtype=_np.float32)
        gram = Xc @ X_T  # (bs, n) on GPU

        if metric == "euclidean":
            sq_c = vnp.sum(Xc * Xc, axis=1).reshape(bs, 1)
            d2 = vnp.maximum(sq_c + sq_all - gram * 2.0, 0.0)
            top_idx = vnp.top_k(-d2, k=k_query, axis=1)
            idx_chunk = top_idx.to_numpy().astype(_np.int32)
            d2_np = d2.to_numpy()
            d_chunk = _np.sqrt(_np.maximum(_np.take_along_axis(d2_np, idx_chunk, axis=1), 0.0))
        else:
            # Cosine similarity: largest dot product = smallest cosine distance (1 - sim)
            top_idx = vnp.top_k(gram, k=k_query, axis=1)
            idx_chunk = top_idx.to_numpy().astype(_np.int32)
            gram_np = gram.to_numpy()
            d_chunk = _np.maximum(1.0 - _np.take_along_axis(gram_np, idx_chunk, axis=1), 0.0)

        # Sort each row by ascending distance (ggml top_k may not guarantee strict order)
        ord_r = _np.argsort(d_chunk, axis=1)
        idx_chunk = _np.take_along_axis(idx_chunk, ord_r, axis=1)
        d_chunk = _np.take_along_axis(d_chunk, ord_r, axis=1)

        # Strip self-index i from each row
        rows = _np.arange(start, end, dtype=_np.int32)[:, None]
        is_self = idx_chunk == rows
        has_self = _np.any(is_self, axis=1)
        for r in range(bs):
            if has_self[r]:
                keep = ~is_self[r]
                out_idx[start + r] = idx_chunk[r, keep][:k_eff]
                out_dist[start + r] = d_chunk[r, keep][:k_eff]
            else:
                out_idx[start + r] = idx_chunk[r, :k_eff]
                out_dist[start + r] = d_chunk[r, :k_eff]

    return out_idx, out_dist


class NNDescent:
    """Approximate K-NN graph builder using GPU-accelerated neighbor-of-neighbor refinement.

    Follows the ``mlx-vis`` ``NNDescent(k=15).build(X)`` API.
    """

    def __init__(
        self,
        k: int = 15,
        metric: str = "euclidean",
        n_iters: int = 10,
        max_candidates: int = 32,
        seed: int = 42,
    ) -> None:
        self.k = int(k)
        self.metric = metric
        self.n_iters = int(n_iters)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)

    def build(self, X) -> Tuple[_np.ndarray, _np.ndarray]:
        """Build approximate K-NN graph ``(indices, distances)`` for ``X``."""
        X_np = _to_f32_2d(X)
        n, d = X_np.shape
        k_eff = min(self.k, max(1, n - 1))

        # For small-to-medium N, exact GPU brute-force is already sub-second & exact
        if n <= 4096:
            return brute_force_knn(X_np, k=k_eff, metric=self.metric)

        if self.metric == "cosine":
            norms = _np.linalg.norm(X_np, axis=1, keepdims=True)
            X_use = X_np / _np.maximum(norms, 1e-8)
        elif self.metric == "euclidean":
            X_use = X_np
        else:
            raise ValueError(f"Unsupported metric '{self.metric}'")

        rng = _np.random.default_rng(self.seed)

        # Initialize with random-projection sorted neighbors + random neighbors
        proj = rng.standard_normal((d, 4)).astype(_np.float32)
        scores = X_use @ proj
        cand_list = []
        step_k = max(2, k_eff // 4)
        for col in range(4):
            order = _np.argsort(scores[:, col])
            inv = _np.empty(n, dtype=_np.int32)
            inv[order] = _np.arange(n, dtype=_np.int32)
            for off in range(1, step_k + 1):
                cand_list.append(order[_np.minimum(inv + off, n - 1)])
                cand_list.append(order[_np.maximum(inv - off, 0)])
        rand_init = rng.integers(0, n, size=(n, k_eff), dtype=_np.int32)
        cands = _np.concatenate([_np.stack(cand_list, axis=1), rand_init], axis=1)

        X_gpu = vnp.array(X_use, dtype=_np.float32)

        def _eval_candidates(cand_idx: _np.ndarray) -> Tuple[_np.ndarray, _np.ndarray]:
            """Evaluate distances for (N, C) candidates on GPU in chunks and keep top-k."""
            c_width = cand_idx.shape[1]
            best_i = _np.empty((n, k_eff), dtype=_np.int32)
            best_d = _np.empty((n, k_eff), dtype=_np.float32)
            chunk = max(256, min(2048, (2_000_000) // max(1, c_width * d)))
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                bs = e - s
                ci = cand_idx[s:e]  # (bs, C)
                Xc = vnp.array(X_use[s:e], dtype=_np.float32).reshape(bs, 1, d)
                Xn = vnp.take(X_gpu, vnp.array(ci.ravel(), dtype=_np.int32), axis=0).reshape(bs, c_width, d)
                if self.metric == "euclidean":
                    diff = Xc - Xn
                    d2 = vnp.sum(diff * diff, axis=2)
                    dc = _np.sqrt(_np.maximum(d2.to_numpy(), 0.0))
                else:
                    sim = vnp.sum(Xc * Xn, axis=2)
                    dc = _np.maximum(1.0 - sim.to_numpy(), 0.0)

                # Mask self-matches
                rows = _np.arange(s, e, dtype=_np.int32)[:, None]
                dc = _np.where(ci == rows, _np.float32(1e30), dc)

                # Deduplicate & pick smallest k_eff per row
                ord_c = _np.argsort(dc, axis=1)
                for r in range(bs):
                    seen = set()
                    cnt = 0
                    for pos in ord_c[r]:
                        nid = int(ci[r, pos])
                        if nid == s + r or nid in seen:
                            continue
                        seen.add(nid)
                        best_i[s + r, cnt] = nid
                        best_d[s + r, cnt] = dc[r, pos]
                        cnt += 1
                        if cnt >= k_eff:
                            break
                    while cnt < k_eff:
                        best_i[s + r, cnt] = (s + r + cnt + 1) % n
                        best_d[s + r, cnt] = 1e6
                        cnt += 1
            return best_i, best_d

        nbr_idx, nbr_dist = _eval_candidates(cands)

        # Iterative local join: neighbors-of-neighbors
        n_prop = min(k_eff, max(4, self.max_candidates // 2))
        for _ in range(self.n_iters):
            # Gather 2-hop neighbors: for each i, take neighbors of its top n_prop neighbors
            sub = nbr_idx[:, :n_prop]  # (N, n_prop)
            two_hop = nbr_idx[sub, :4].reshape(n, n_prop * 4)
            # Also add a few random probes for global connectivity
            probes = rng.integers(0, n, size=(n, 4), dtype=_np.int32)
            all_cands = _np.concatenate([nbr_idx, two_hop, probes], axis=1)
            new_idx, new_dist = _eval_candidates(all_cands)
            improved = _np.mean(new_dist < nbr_dist - 1e-6)
            nbr_idx, nbr_dist = new_idx, new_dist
            if improved < 0.002:
                break

        return nbr_idx, nbr_dist


def knn_search(
    X,
    k: int = 15,
    metric: str = "euclidean",
    method: str = "auto",
    seed: int = 42,
) -> Tuple[_np.ndarray, _np.ndarray]:
    """Unified K-NN entry point (matches ``mlx-vis`` ``knn_method`` convention)."""
    X_np = _to_f32_2d(X)
    n = X_np.shape[0]
    if method == "auto":
        use_method = "brute" if n <= 20000 else "nndescent"
    else:
        use_method = method

    if use_method == "brute":
        return brute_force_knn(X_np, k=k, metric=metric)
    if use_method == "nndescent":
        return NNDescent(k=k, metric=metric, seed=seed).build(X_np)
    raise ValueError(f"Unknown knn_method '{method}'. Use 'auto', 'brute', or 'nndescent'.")


def symmetrize_knn_table(
    indices: _np.ndarray,
    weights: _np.ndarray,
    max_deg: int | None = None,
    mode: str = "fuzzy",
) -> Tuple[_np.ndarray, _np.ndarray]:
    """Build a fixed-degree symmetric neighbor table ``(sym_idx, sym_w)`` of shape ``(N, K_sym)``.

    Having every node hold its own fixed-size symmetric neighbor list enables
    zero-scatter, coalesced ``vnp.take`` + ``vnp.sum(..., axis=1)`` gradient
    accumulation on the Vulkan GPU.
    """
    n, k = indices.shape
    k_sym = int(max_deg) if max_deg is not None else min(n - 1, max(k, int(k * 2)))

    # Accumulate directed edges and reverse edges in Python dicts only for top-k_sym
    # Fast vectorized reverse edge sort:
    src = _np.repeat(_np.arange(n, dtype=_np.int32), k)
    dst = indices.ravel().astype(_np.int32)
    w = weights.ravel().astype(_np.float32)

    # Combine (src -> dst, w) and (dst -> src, w)
    all_u = _np.concatenate([src, dst])
    all_v = _np.concatenate([dst, src])
    all_w = _np.concatenate([w, w])

    # Sort by (u, v) to merge duplicate bidirectional edges
    order = _np.lexsort((all_v, all_u))
    all_u = all_u[order]
    all_v = all_v[order]
    all_w = all_w[order]

    # Identify unique (u, v) pairs
    same = (all_u[1:] == all_u[:-1]) & (all_v[1:] == all_v[:-1])
    keep = _np.concatenate([[True], ~same])
    uniq_u = all_u[keep]
    uniq_v = all_v[keep]
    uniq_w = all_w[keep].copy()

    # For duplicates (bidirectional edges), apply symmetric merge formula
    dup_pos = _np.nonzero(same)[0]
    if dup_pos.size > 0:
        # Map duplicate back to its kept slot (immediately preceding kept entry)
        kept_idx = _np.cumsum(keep) - 1
        w1 = uniq_w[kept_idx[dup_pos]]
        w2 = all_w[dup_pos + 1]
        if mode == "fuzzy":
            uniq_w[kept_idx[dup_pos]] = w1 + w2 - w1 * w2
        else:
            uniq_w[kept_idx[dup_pos]] = 0.5 * (w1 + w2)

    # Sort within each u by descending weight and pack into (N, k_sym)
    ord_w = _np.lexsort((-uniq_w, uniq_u))
    uniq_u = uniq_u[ord_w]
    uniq_v = uniq_v[ord_w]
    uniq_w = uniq_w[ord_w]

    sym_idx = _np.zeros((n, k_sym), dtype=_np.int32)
    sym_w = _np.zeros((n, k_sym), dtype=_np.float32)

    counts = _np.bincount(uniq_u, minlength=n)
    starts = _np.zeros(n + 1, dtype=_np.int64)
    _np.cumsum(counts, out=starts[1:])

    for i in range(n):
        s, e = int(starts[i]), int(starts[i + 1])
        m = min(e - s, k_sym)
        if m > 0:
            sym_idx[i, :m] = uniq_v[s : s + m]
            sym_w[i, :m] = uniq_w[s : s + m]
            if m < k_sym:
                sym_idx[i, m:] = i  # padded self-index with weight 0
        else:
            sym_idx[i, :] = i

    return sym_idx, sym_w
