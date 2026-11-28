"""XGBoost-style 2nd-order Histogram Gradient Boosting (XGBRegressor & XGBClassifier) for vklearn.

Implements the exact XGBoost ``tree_method='hist'`` algorithm:
- 2nd-order Newton-Raphson gradients (g) & Hessians (h)
- L2 leaf regularization (``reg_lambda``) & split-loss pruning (``gamma``)
- Native ``NaN`` missing-value default-direction learning at every split
- Row subsampling (``subsample``) & column subsampling (``colsample_bytree``)
"""

from __future__ import annotations

from typing import Any, List, Optional
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_1d, to_gpu_2d
from .metrics import accuracy_score, r2_score


class _XGBNode:
    __slots__ = ("feature", "threshold", "default_left", "left", "right", "weight", "gain")

    def __init__(
        self,
        *,
        weight: float = 0.0,
        feature: int = -1,
        threshold: float = 0.0,
        default_left: bool = True,
        left: Optional["_XGBNode"] = None,
        right: Optional["_XGBNode"] = None,
        gain: float = 0.0,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.default_left = default_left
        self.left = left
        self.right = right
        self.weight = weight
        self.gain = gain


def _xgb_bin_edges(X: np.ndarray, max_bins: int) -> List[np.ndarray]:
    _, d = X.shape
    edges: List[np.ndarray] = []
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    for j in range(d):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) <= 1:
            edges.append(np.empty(0, dtype=np.float32))
            continue
        u = np.unique(valid)
        if len(u) <= max_bins:
            th = ((u[:-1] + u[1:]) * 0.5).astype(np.float32) if len(u) > 1 else np.empty(0, dtype=np.float32)
        else:
            th = np.unique(np.quantile(valid, qs)).astype(np.float32)
        edges.append(th)
    return edges


class _XGBTree:
    """Single 2nd-order regularized histogram tree with native NaN routing."""

    def __init__(
        self,
        max_depth: int = 6,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        colsample_bytree: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.colsample_bytree = float(colsample_bytree)
        self.rng = rng or np.random.default_rng(42)
        self.root: Optional[_XGBNode] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        idx: np.ndarray,
        edges: List[np.ndarray],
    ) -> "_XGBTree":
        d = X.shape[1]
        self.feature_importances_ = np.zeros(d, dtype=np.float64)
        k_cols = max(1, min(d, int(round(d * self.colsample_bytree))))
        feats = (
            self.rng.choice(d, size=k_cols, replace=False)
            if k_cols < d
            else np.arange(d, dtype=np.int32)
        )
        self.root = self._build(X, g, h, idx, edges, feats, depth=0)
        return self

    def _build(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        idx: np.ndarray,
        edges: List[np.ndarray],
        feats: np.ndarray,
        depth: int,
    ) -> _XGBNode:
        gs = g[idx]
        hs = h[idx]
        G_tot = float(np.sum(gs))
        H_tot = float(np.sum(hs))
        leaf_w = -G_tot / (H_tot + self.reg_lambda)

        if depth >= self.max_depth or len(idx) < 2 or H_tot < 2.0 * self.min_child_weight:
            return _XGBNode(weight=leaf_w)

        parent_score = (G_tot * G_tot) / (H_tot + self.reg_lambda)
        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        best_default_left = True

        lam = self.reg_lambda
        mcw = self.min_child_weight

        for j in feats:
            th = edges[int(j)]
            if len(th) == 0:
                continue
            col = X[idx, int(j)]
            nan_mask = np.isnan(col)
            has_nan = bool(np.any(nan_mask))

            if has_nan:
                valid_mask = ~nan_mask
                if not np.any(valid_mask):
                    continue
                G_nan = float(np.sum(gs[nan_mask]))
                H_nan = float(np.sum(hs[nan_mask]))
                v_col = col[valid_mask]
                v_g = gs[valid_mask]
                v_h = hs[valid_mask]
            else:
                G_nan = 0.0
                H_nan = 0.0
                v_col = col
                v_g = gs
                v_h = hs

            bins = np.searchsorted(th, v_col, side="right")
            nb = len(th) + 1
            G_hist = np.bincount(bins, weights=v_g, minlength=nb)
            H_hist = np.bincount(bins, weights=v_h, minlength=nb)
            GL_base = np.cumsum(G_hist[:-1])
            HL_base = np.cumsum(H_hist[:-1])
            GR_base = (G_tot - G_nan) - GL_base
            HR_base = (H_tot - H_nan) - HL_base

            # Option A: NaNs go left; Option B: NaNs go right
            candidates = (
                ((GL_base + G_nan, HL_base + H_nan, GR_base, HR_base, True),
                 (GL_base, HL_base, GR_base + G_nan, HR_base + H_nan, False))
                if has_nan
                else ((GL_base, HL_base, GR_base, HR_base, True),)
            )
            for GL, HL, GR, HR, def_left in candidates:
                ok = (HL >= mcw) & (HR >= mcw)
                if not np.any(ok):
                    continue
                gains = np.full(len(th), -np.inf, dtype=np.float64)
                gains[ok] = 0.5 * (
                    (GL[ok] ** 2) / (HL[ok] + lam)
                    + (GR[ok] ** 2) / (HR[ok] + lam)
                    - parent_score
                ) - self.gamma
                b_i = int(np.argmax(gains))
                if gains[b_i] > best_gain:
                    best_gain = float(gains[b_i])
                    best_feat = int(j)
                    best_thresh = float(th[b_i])
                    best_default_left = def_left

        if best_feat == -1 or best_gain <= 1e-12:
            return _XGBNode(weight=leaf_w)

        assert self.feature_importances_ is not None
        self.feature_importances_[best_feat] += best_gain

        col_best = X[idx, best_feat]
        is_nan = np.isnan(col_best)
        go_left = (col_best <= best_thresh) | (is_nan if best_default_left else False)
        if not best_default_left:
            go_left = go_left & (~is_nan)
        left_idx = idx[go_left]
        right_idx = idx[~go_left]
        if len(left_idx) == 0 or len(right_idx) == 0:
            return _XGBNode(weight=leaf_w)

        return _XGBNode(
            feature=best_feat,
            threshold=best_thresh,
            default_left=best_default_left,
            left=self._build(X, g, h, left_idx, edges, feats, depth + 1),
            right=self._build(X, g, h, right_idx, edges, feats, depth + 1),
            gain=best_gain,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        out = np.empty(X.shape[0], dtype=np.float64)

        def _eval(node: _XGBNode, idx: np.ndarray) -> None:
            if len(idx) == 0:
                return
            if node.feature == -1:
                out[idx] = node.weight
                return
            col = X[idx, node.feature]
            is_nan = np.isnan(col)
            go_left = (col <= node.threshold) | (is_nan if node.default_left else False)
            if not node.default_left:
                go_left = go_left & (~is_nan)
            _eval(node.left, idx[go_left])  # type: ignore[arg-type]
            _eval(node.right, idx[~go_left])  # type: ignore[arg-type]

        _eval(self.root, np.arange(X.shape[0], dtype=np.int32))
        return out


class XGBRegressor:
    """XGBoost-style 2nd-order Histogram Gradient Boosted Regressor with native NaN routing."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        max_bins: int = 64,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.base_score_: float = 0.0
        self.trees_: List[_XGBTree] = []
        self.feature_importances_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any) -> "XGBRegressor":
        X_np = to_gpu_2d(X).to_numpy().astype(np.float32)
        y_np = to_gpu_1d(y).to_numpy().astype(np.float64)
        n, d = X_np.shape
        edges = _xgb_bin_edges(X_np, self.max_bins)
        rng = np.random.default_rng(self.random_state)

        self.base_score_ = float(np.mean(y_np))
        pred = np.full(n, self.base_score_, dtype=np.float64)
        h_ones = np.ones(n, dtype=np.float64)
        all_idx = np.arange(n, dtype=np.int32)
        sub_n = max(2, int(round(n * self.subsample)))

        self.trees_ = []
        imp = np.zeros(d, dtype=np.float64)
        for _ in range(self.n_estimators):
            g = pred - y_np  # 1st derivative of 0.5*(pred - y)^2
            idx = rng.choice(n, size=sub_n, replace=False).astype(np.int32) if self.subsample < 1.0 else all_idx
            tree = _XGBTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                colsample_bytree=self.colsample_bytree,
                rng=rng,
            ).fit(X_np, g, h_ones, idx, edges)
            pred += self.learning_rate * tree.predict(X_np)
            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                imp += tree.feature_importances_

        tot = float(np.sum(imp))
        self.feature_importances_ = vnp.array((imp / tot if tot > 0 else imp).astype(np.float32))
        return self

    def predict(self, X: Any) -> vnp.ndarray:
        if not self.trees_:
            raise RuntimeError("XGBRegressor must be fitted before predict()")
        X_np = to_gpu_2d(X).to_numpy().astype(np.float32)
        pred = np.full(X_np.shape[0], self.base_score_, dtype=np.float64)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X_np)
        return vnp.array(pred.astype(np.float32))

    def score(self, X: Any, y: Any) -> float:
        return r2_score(y, self.predict(X))


class XGBClassifier:
    """XGBoost-style 2nd-order Histogram Gradient Boosted Classifier (binary & multiclass, native NaN)."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        max_bins: int = 64,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.init_logits_: Optional[np.ndarray] = None
        self.trees_: List[List[_XGBTree]] = []
        self.feature_importances_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any) -> "XGBClassifier":
        X_np = to_gpu_2d(X).to_numpy().astype(np.float32)
        y_raw = to_gpu_1d(y).to_numpy().astype(np.int32)
        classes, y_codes = np.unique(y_raw, return_inverse=True)
        self.classes_ = classes
        k = len(classes)
        n, d = X_np.shape
        edges = _xgb_bin_edges(X_np, self.max_bins)
        rng = np.random.default_rng(self.random_state)

        priors = np.bincount(y_codes, minlength=k).astype(np.float64) / max(n, 1)
        self.init_logits_ = np.log(np.clip(priors, 1e-6, 1.0))
        logits = np.broadcast_to(self.init_logits_[None, :], (n, k)).copy()
        y_oh = np.eye(k, dtype=np.float64)[y_codes]

        all_idx = np.arange(n, dtype=np.int32)
        sub_n = max(2, int(round(n * self.subsample)))
        self.trees_ = []
        imp = np.zeros(d, dtype=np.float64)

        for _ in range(self.n_estimators):
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            idx = rng.choice(n, size=sub_n, replace=False).astype(np.int32) if self.subsample < 1.0 else all_idx

            step_trees: List[_XGBTree] = []
            for c in range(k):
                p_c = probs[:, c]
                g_c = p_c - y_oh[:, c]
                h_c = np.maximum(p_c * (1.0 - p_c), 1e-6)
                tree = _XGBTree(
                    max_depth=self.max_depth,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                    min_child_weight=self.min_child_weight,
                    colsample_bytree=self.colsample_bytree,
                    rng=rng,
                ).fit(X_np, g_c, h_c, idx, edges)
                logits[:, c] += self.learning_rate * tree.predict(X_np)
                step_trees.append(tree)
                if tree.feature_importances_ is not None:
                    imp += tree.feature_importances_
            self.trees_.append(step_trees)

        tot = float(np.sum(imp))
        self.feature_importances_ = vnp.array((imp / tot if tot > 0 else imp).astype(np.float32))
        return self

    def predict_proba(self, X: Any) -> vnp.ndarray:
        if not self.trees_ or self.classes_ is None or self.init_logits_ is None:
            raise RuntimeError("XGBClassifier must be fitted before predict_proba()")
        X_np = to_gpu_2d(X).to_numpy().astype(np.float32)
        k = len(self.classes_)
        logits = np.broadcast_to(self.init_logits_[None, :], (X_np.shape[0], k)).copy()
        for step_trees in self.trees_:
            for c, tree in enumerate(step_trees):
                logits[:, c] += self.learning_rate * tree.predict(X_np)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        return vnp.array(probs.astype(np.float32))

    def predict(self, X: Any) -> vnp.ndarray:
        assert self.classes_ is not None
        probs = self.predict_proba(X).to_numpy()
        return vnp.array(self.classes_[np.argmax(probs, axis=1)].astype(np.int32), dtype=np.int32)

    def score(self, X: Any, y: Any) -> float:
        return accuracy_score(y, self.predict(X))


HistGradientBoostingRegressor = XGBRegressor
HistGradientBoostingClassifier = XGBClassifier

__all__ = [
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "XGBClassifier",
    "XGBRegressor",
]
