"""Tests for cuML-style tabular ML models (vklearn) and cuVS-style IVFIndex (vkvec)."""

from __future__ import annotations

import numpy as np

import vklearn as vkl
import vknumpy as vnp
import vkvec


def test_knn_regressor_and_dbscan_gpu():
    rng = np.random.default_rng(7)
    X_reg = rng.uniform(-3.0, 3.0, size=(200, 2)).astype(np.float32)
    y_reg = (np.sin(X_reg[:, 0]) + 0.5 * X_reg[:, 1]).astype(np.float32)

    knn_u = vkl.KNeighborsRegressor(n_neighbors=5, weights="uniform").fit(X_reg[:160], y_reg[:160])
    knn_d = vkl.KNeighborsRegressor(n_neighbors=5, weights="distance").fit(X_reg[:160], y_reg[:160])
    assert knn_u.score(X_reg[160:], y_reg[160:]) > 0.90
    assert knn_d.score(X_reg[160:], y_reg[160:]) > 0.92

    # Two dense clusters + 3 far noise outliers
    c0 = rng.normal(loc=[-4.0, -4.0], scale=0.25, size=(40, 2)).astype(np.float32)
    c1 = rng.normal(loc=[ 4.0,  4.0], scale=0.25, size=(40, 2)).astype(np.float32)
    noise = np.array([[20.0, 20.0], [-20.0, 15.0], [0.0, -25.0]], dtype=np.float32)
    X_db = np.vstack([c0, c1, noise])
    labels = vkl.DBSCAN(eps=1.0, min_samples=5).fit_predict(X_db).tolist()
    assert len(set(labels[:40])) == 1 and labels[0] >= 0
    assert len(set(labels[40:80])) == 1 and labels[40] >= 0 and labels[40] != labels[0]
    assert labels[80:] == [-1, -1, -1]


def test_random_forest_and_gradient_boosting_and_cv():
    rng = np.random.default_rng(42)

    # Non-linear regression: y = x0^2 - 2*x1 + noise
    X_r = rng.uniform(-2.0, 2.0, size=(300, 4)).astype(np.float32)
    y_r = (X_r[:, 0] ** 2 - 2.0 * X_r[:, 1] + rng.normal(0, 0.1, 300)).astype(np.float32)
    X_tr, X_te, y_tr, y_te = vkl.train_test_split(X_r, y_r, test_size=0.25, random_state=42)

    rf_r = vkl.RandomForestRegressor(n_estimators=25, max_depth=6, random_state=42).fit(X_tr, y_tr)
    gb_r = vkl.GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=3, random_state=42).fit(X_tr, y_tr)
    assert rf_r.score(X_te, y_te) > 0.85
    assert gb_r.score(X_te, y_te) > 0.90
    assert rf_r.feature_importances_ is not None and abs(float(vnp.sum(rf_r.feature_importances_)) - 1.0) < 1e-3
    # Features 0 and 1 should dominate over noise features 2 and 3
    imp_gb = gb_r.feature_importances_.tolist()
    assert imp_gb[0] + imp_gb[1] > 0.85

    # 3-class classification
    X_c = np.vstack([
        rng.normal([-2.5, 0.0, 0.0], 0.5, (60, 3)),
        rng.normal([ 2.5, 0.0, 0.0], 0.5, (60, 3)),
        rng.normal([ 0.0, 2.8, 0.0], 0.5, (60, 3)),
    ]).astype(np.float32)
    y_c = np.array([0] * 60 + [1] * 60 + [2] * 60, dtype=np.int32)
    Xc_tr, Xc_te, yc_tr, yc_te = vkl.train_test_split(X_c, y_c, test_size=0.25, random_state=42)

    rf_c = vkl.RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42).fit(Xc_tr, yc_tr)
    gb_c = vkl.GradientBoostingClassifier(n_estimators=40, learning_rate=0.15, max_depth=3, random_state=42).fit(Xc_tr, yc_tr)
    assert rf_c.score(Xc_te, yc_te) > 0.93
    assert gb_c.score(Xc_te, yc_te) > 0.93
    assert rf_c.predict_proba(Xc_te).shape == (len(yc_te), 3)
    assert gb_c.predict_proba(Xc_te).shape == (len(yc_te), 3)

    # K-fold cross_val_score
    cv_scores = vkl.cross_val_score(
        vkl.RandomForestClassifier(n_estimators=15, max_depth=5, random_state=0),
        X_c, y_c, cv=4,
    )
    assert len(cv_scores) == 4
    assert float(vnp.mean(cv_scores)) > 0.92


def test_vklearn_accel_ensemble():
    vkl.accel.install()
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, (80, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        clf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=1).fit(X, y)
        assert clf.score(X, y) > 0.90

        reg = GradientBoostingRegressor(n_estimators=25, max_depth=3, random_state=1).fit(X, X[:, 0] * 2.0)
        assert reg.score(X, X[:, 0] * 2.0) > 0.90
    finally:
        vkl.accel.uninstall()


def test_vkvec_ivf_index_cosine_l2_filter_and_persistence(tmp_path):
    rng = np.random.default_rng(99)
    # 600 clustered 32-D vectors across 12 topic centers
    centers = rng.normal(0, 1, (12, 32)).astype(np.float32)
    vecs = np.vstack([c + rng.normal(0, 0.15, (50, 32)) for c in centers]).astype(np.float32)
    meta = [{"topic": i // 50, "idx": i} for i in range(len(vecs))]

    ivf = vkvec.IVFIndex(nlist=12, nprobe=4, metric="cosine", seed=42)
    ivf.add(vecs, metadata=meta)
    assert len(ivf) == 600
    assert ivf.is_trained

    # Compare recall against exact brute-force VectorIndex
    exact = vkvec.VectorIndex(metric="cosine").add(vecs, metadata=meta)
    hits_ivf = ivf.search(vecs[5], k=10)
    hits_exact = exact.search(vecs[5], k=10)[0]
    assert hits_ivf[0]["id"] == 5
    overlap = len({h["id"] for h in hits_ivf} & {h["id"] for h in hits_exact})
    assert overlap >= 8

    # Metadata filter
    filtered = ivf.search(vecs[5], k=5, nprobe=12, where={"topic": 3})
    assert len(filtered) == 5
    assert all(h["metadata"]["topic"] == 3 for h in filtered)

    # Save & reload roundtrip
    save_path = tmp_path / "demo_ivf"
    ivf.save(save_path)
    loaded = vkvec.IVFIndex.load(save_path)
    assert len(loaded) == 600
    hits_loaded = loaded.search(vecs[5], k=5)
    assert [h["id"] for h in hits_loaded] == [h["id"] for h in hits_ivf[:5]]

    # L2 metric test
    ivf_l2 = vkvec.IVFIndex(nlist=8, nprobe=4, metric="l2", seed=42).add(vecs)
    l2_hits = ivf_l2.search(vecs[100], k=3)
    assert l2_hits[0]["id"] == 100
    assert l2_hits[0]["score"] < 1e-3
