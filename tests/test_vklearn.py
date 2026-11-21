import numpy as np
import vknumpy as vnp
import vkpandas as vpd
import vklearn as vkl


def test_scalers_and_pipeline_regression():
    rng = np.random.default_rng(42)
    X_np = rng.normal(loc=[10.0, -5.0, 2.0], scale=[3.0, 0.5, 4.0], size=(4000, 3)).astype(np.float32)
    true_w = np.array([2.5, -1.2, 0.7], dtype=np.float32)
    y_np = X_np @ true_w + 4.0 + rng.normal(0.0, 0.05, size=4000).astype(np.float32)

    df = vpd.DataFrame({"f0": X_np[:, 0], "f1": X_np[:, 1], "f2": X_np[:, 2]})
    X_tr, X_te, y_tr, y_te = vkl.train_test_split(df, vnp.array(y_np), test_size=0.2, random_state=7)

    pipe = vkl.Pipeline([
        ("scaler", vkl.StandardScaler()),
        ("reg", vkl.LinearRegression()),
    ])
    pipe.fit(X_tr, y_tr)
    r2 = pipe.score(X_te, y_te)
    assert r2 > 0.999

    # MinMaxScaler bounds check
    mm = vkl.MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X_tr)
    assert float(vnp.min(mm).item()) >= -1.001
    assert float(vnp.max(mm).item()) <= 1.001

    # Ridge direct fit recovers true coefficients
    ridge = vkl.Ridge(alpha=1e-3).fit(X_tr, y_tr)
    np.testing.assert_allclose(ridge.coef_.to_numpy(), true_w, atol=0.05)
    assert abs(ridge.intercept_ - 4.0) < 0.05


def test_logistic_regression_and_metrics():
    rng = np.random.default_rng(123)
    n = 3000
    X0 = rng.normal(-2.0, 0.8, size=(n // 2, 4)).astype(np.float32)
    X1 = rng.normal(2.0, 0.8, size=(n // 2, 4)).astype(np.float32)
    X = vnp.array(np.vstack([X0, X1]))
    y = vnp.array(np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int32))

    X_tr, X_te, y_tr, y_te = vkl.train_test_split(X, y, test_size=0.25, random_state=0)
    clf = vkl.LogisticRegression(C=1.0).fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    assert acc > 0.98
    assert clf.n_iter_ <= 15  # Newton-IRLS converges quadratically

    cm = vkl.confusion_matrix(y_te, clf.predict(X_te), num_classes=2).to_numpy()
    assert cm.shape == (2, 2)
    assert int(cm.sum()) == len(y_te)


def test_pca_reconstruction():
    rng = np.random.default_rng(9)
    Z = rng.normal(0.0, 1.0, size=(2000, 2)).astype(np.float32)
    W = np.array([[1.0, 0.5, -0.3, 0.8], [-0.4, 1.2, 0.9, -0.2]], dtype=np.float32)
    X = Z @ W + np.array([3.0, -1.0, 2.0, 0.5], dtype=np.float32)

    pca = vkl.PCA(n_components=2).fit(X)
    assert float(pca.explained_variance_ratio_.sum()) > 0.999
    X_rec = pca.inverse_transform(pca.transform(X)).to_numpy()
    np.testing.assert_allclose(X_rec, X, atol=1e-3)


def test_kmeans_and_knn():
    rng = np.random.default_rng(5)
    centers_true = np.array([[-6.0, -6.0], [6.0, -6.0], [0.0, 6.0]], dtype=np.float32)
    pts = np.vstack([c + rng.normal(0.0, 0.4, size=(600, 2)).astype(np.float32) for c in centers_true])
    true_lbl = np.repeat(np.arange(3, dtype=np.int32), 600)

    km = vkl.KMeans(n_clusters=3, random_state=42).fit(pts)
    assert km.cluster_centers_.shape == (3, 2)
    # Every true center should be close to one found center
    found = km.cluster_centers_.to_numpy()
    for tc in centers_true:
        assert float(np.min(np.linalg.norm(found - tc, axis=1))) < 0.15

    X_tr, X_te, y_tr, y_te = vkl.train_test_split(vnp.array(pts), vnp.array(true_lbl), test_size=0.2, random_state=1)
    knn = vkl.KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    assert knn.score(X_te, y_te) > 0.99
