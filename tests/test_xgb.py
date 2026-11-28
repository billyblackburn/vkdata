"""Tests for vklearn.xgb: XGBRegressor, XGBClassifier, HistGradientBoosting aliases, and native NaN routing."""

import numpy as np
import vknumpy as vnp
import vklearn as vkl


def test_xgb_regressor_nonlinear_and_regularization():
    rng = np.random.default_rng(7)
    X_np = rng.uniform(-3.0, 3.0, size=(300, 4)).astype(np.float32)
    y_np = (np.sin(X_np[:, 0]) * 2.0 + 0.5 * X_np[:, 1] ** 2 - 1.2 * X_np[:, 2]).astype(np.float32)

    X_tr, X_te = vnp.array(X_np[:240]), vnp.array(X_np[240:])
    y_tr, y_te = vnp.array(y_np[:240]), vnp.array(y_np[240:])

    model = vkl.XGBRegressor(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=1.5,
        gamma=0.01,
        subsample=0.85,
        colsample_bytree=0.9,
        random_state=42,
    ).fit(X_tr, y_tr)

    r2 = model.score(X_te, y_te)
    assert r2 > 0.90, f"Expected R² > 0.90, got {r2:.4f}"
    imp = model.feature_importances_.to_numpy()
    assert imp.shape == (4,)
    assert abs(float(np.sum(imp)) - 1.0) < 1e-4
    # Noise feature X[:, 3] should have much lower importance than signal features
    assert imp[3] < imp[0] and imp[3] < imp[1]


def test_xgb_regressor_native_nan_routing():
    rng = np.random.default_rng(19)
    X_np = rng.normal(size=(260, 3)).astype(np.float32)
    y_np = (3.0 * X_np[:, 0] - 2.0 * X_np[:, 1] + 0.5 * X_np[:, 2] ** 2).astype(np.float32)

    # Inject 15% NaNs into both train and test without any imputation
    mask = rng.random(X_np.shape) < 0.15
    X_missing = X_np.copy()
    X_missing[mask] = np.nan

    X_tr, X_te = vnp.array(X_missing[:200]), vnp.array(X_missing[200:])
    y_tr, y_te = vnp.array(y_np[:200]), vnp.array(y_np[200:])

    model = vkl.XGBRegressor(
        n_estimators=70,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=1.0,
        random_state=0,
    ).fit(X_tr, y_tr)

    preds = model.predict(X_te).to_numpy()
    assert not np.any(np.isnan(preds)), "Predictions must not contain NaNs"
    r2 = model.score(X_te, y_te)
    assert r2 > 0.70, f"Expected R² > 0.70 with 15% missing values, got {r2:.4f}"


def test_xgb_classifier_multiclass_and_native_nan():
    rng = np.random.default_rng(11)
    n_per = 80
    c0 = rng.normal(loc=[-3.0, -2.0, 0.0], scale=0.8, size=(n_per, 3))
    c1 = rng.normal(loc=[3.0, -1.0, 1.0], scale=0.8, size=(n_per, 3))
    c2 = rng.normal(loc=[0.0, 3.5, -1.5], scale=0.8, size=(n_per, 3))
    X_np = np.vstack([c0, c1, c2]).astype(np.float32)
    y_np = np.array([0] * n_per + [1] * n_per + [2] * n_per, dtype=np.int32)
    perm = rng.permutation(len(y_np))
    X_np, y_np = X_np[perm], y_np[perm]

    # Inject 10% NaNs
    X_np[rng.random(X_np.shape) < 0.10] = np.nan

    X_tr, X_te = vnp.array(X_np[:180]), vnp.array(X_np[180:])
    y_tr, y_te = vnp.array(y_np[:180]), vnp.array(y_np[180:])

    clf = vkl.XGBClassifier(
        n_estimators=50,
        learning_rate=0.12,
        max_depth=4,
        reg_lambda=1.0,
        gamma=0.05,
        colsample_bytree=0.8,
        random_state=42,
    ).fit(X_tr, y_tr)

    probs = clf.predict_proba(X_te).to_numpy()
    assert probs.shape == (60, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    acc = clf.score(X_te, y_te)
    assert acc > 0.90, f"Expected multiclass accuracy > 0.90 with 10% NaNs, got {acc:.4f}"


def test_hist_gradient_boosting_aliases_and_accel():
    assert vkl.HistGradientBoostingRegressor is vkl.XGBRegressor
    assert vkl.HistGradientBoostingClassifier is vkl.XGBClassifier
    assert vkl.ensemble.XGBRegressor is vkl.XGBRegressor
    assert vkl.ensemble.HistGradientBoostingClassifier is vkl.XGBClassifier

    vkl.accel.install()
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, XGBRegressor  # type: ignore

        X = np.vstack([
            np.zeros((8, 2), dtype=np.float32),
            np.full((8, 2), 5.0, dtype=np.float32),
        ])
        y = np.array([0] * 8 + [1] * 8, dtype=np.int32)
        clf = HistGradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0).fit(X, y)
        assert clf.score(X, y) == 1.0

        y_reg = np.array([1.0] * 8 + [10.0] * 8, dtype=np.float32)
        reg = XGBRegressor(n_estimators=20, max_depth=2, random_state=0).fit(X, y_reg)
        assert reg.score(X, y_reg) > 0.95
    finally:
        vkl.accel.uninstall()
