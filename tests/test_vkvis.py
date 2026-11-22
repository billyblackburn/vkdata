"""Unit tests for vkvis (Vulkan GPU dimensionality reduction & neighbor embedding)."""

import numpy as np
import pytest

import vkvis as vv


def _make_clustered(n_clusters=4, per_cluster=45, dim=16, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 4.0
    X = np.vstack([centers[c] + rng.standard_normal((per_cluster, dim)).astype(np.float32) * 0.6 for c in range(n_clusters)])
    y = np.repeat(np.arange(n_clusters, dtype=np.int32), per_cluster)
    return X, y


def test_knn_brute_force_and_nndescent():
    X, y = _make_clustered(n_clusters=3, per_cluster=40, dim=12, seed=7)
    idx_euc, dist_euc = vv.brute_force_knn(X, k=8, metric="euclidean")
    assert idx_euc.shape == (120, 8)
    assert dist_euc.shape == (120, 8)
    assert np.all(dist_euc >= 0.0)
    # Self should never appear in k-NN
    assert not np.any(idx_euc == np.arange(120)[:, None])
    # Top neighbors should belong to same cluster (>95%)
    same_cluster = np.mean(y[idx_euc] == y[:, None])
    assert same_cluster > 0.95

    idx_cos, dist_cos = vv.brute_force_knn(X, k=8, metric="cosine")
    assert idx_cos.shape == (120, 8)
    assert np.mean(y[idx_cos] == y[:, None]) > 0.90

    idx_nn, dist_nn = vv.NNDescent(k=8, seed=7).build(X)
    assert idx_nn.shape == (120, 8)
    assert np.mean(y[idx_nn] == y[:, None]) > 0.90


def test_all_vkvis_embeddings_and_rendering(tmp_path):
    X, y = _make_clustered(n_clusters=4, per_cluster=35, dim=16, seed=11)

    models = {
        "UMAP": vv.UMAP(n_components=2, n_neighbors=10, n_epochs=40, seed=11),
        "TSNE": vv.TSNE(n_components=2, perplexity=15.0, n_iters=60, seed=11),
        "PaCMAP": vv.PaCMAP(n_components=2, n_neighbors=8, n_iters=(20, 20, 30), seed=11),
        "LocalMAP": vv.LocalMAP(n_components=2, n_neighbors=8, n_iters=(20, 20, 30), seed=11),
        "TriMap": vv.TriMap(n_components=2, n_inliers=6, n_outliers=3, n_random=2, n_iters=45, seed=11),
        "DREAMS": vv.DREAMS(n_components=2, perplexity=15.0, lam=0.15, n_iters=60, seed=11),
        "CNE_infonce": vv.CNE(n_components=2, n_neighbors=10, loss="infonce", n_epochs=40, seed=11),
        "CNE_neg": vv.CNE(n_components=2, n_neighbors=10, loss="neg", n_epochs=40, seed=11),
    }

    embs = {}
    for name, m in models.items():
        Y = m.fit_transform(X, record_every=20 if name == "UMAP" else 0)
        assert Y.shape == (len(X), 2)
        assert Y.dtype == np.float32
        assert np.isfinite(Y).all(), f"{name} produced non-finite values"
        embs[name] = Y

    # Trajectory recording & GIF export
    assert len(models["UMAP"].history_) >= 3
    gif_path = tmp_path / "umap.gif"
    frames = vv.animate_embedding(models["UMAP"].history_, labels=y, width=160, height=120, save_path=gif_path)
    assert len(frames) == len(models["UMAP"].history_)
    assert gif_path.is_file() and gif_path.stat().st_size > 100

    # 2D comparison grid & 3D UMAP sphere render
    grid = vv.compare_embeddings({"UMAP": embs["UMAP"], "PaCMAP": embs["PaCMAP"]}, labels=y, width=160, height=120)
    assert grid.size[0] > 300 and grid.size[1] > 120

    Y3 = vv.UMAP(n_components=3, n_neighbors=10, n_epochs=25, seed=11).fit_transform(X)
    assert Y3.shape == (len(X), 3) and np.isfinite(Y3).all()
    im3 = vv.render_embedding(Y3, labels=y, width=200, height=160, title="3D UMAP")
    assert im3.size == (200, 184)
