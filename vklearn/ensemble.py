"""Histogram-binned Tree Ensembles (RandomForest & GradientBoosting) for vklearn."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np
import vknumpy as vnp

from ._utils import to_gpu_1d, to_gpu_2d
from .metrics import accuracy_score, r2_score


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "gain")

    def __init__(
        self,
        *,
        value: Any = None,
        feature: int = -1,
        threshold: float = 0.0,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
        gain: float = 0.0,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain


def _compute_bin_edges(X: np.ndarray, max_bins: int) -> List[np.ndarray]:
    _, d = X.shape
    edges: List[np.ndarray] = []
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    for j in range(d):
        col = X[:, j]
        u = np.unique(col)
        if len(u) <= max_bins:
            th = (u[:-1] + u[1:]) * 0.5 if len(u) > 1 else np.empty(0, dtype=np.float32)
        else:
            th = np.unique(np.quantile(col, qs)).astype(np.float32)
        edges.append(th)
    return edges


class _HistTreeRegressor:
    """Fast histogram-binned CART regression tree."""

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        max_bins: int = 32,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_bins = int(max_bins)
        self.rng = rng or np.random.default_rng(42)
        self.root: Optional[_Node] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, bin_edges: Optional[List[np.ndarray]] = None) -> "_HistTreeRegressor":
        n, d = X.shape
        self.feature_importances_ = np.zeros(d, dtype=np.float64)
        edges = bin_edges if bin_edges is not None else _compute_bin_edges(X, self.max_bins)
        idx = np.arange(n, dtype=np.int32)
        self.root = self._build(X, y, idx, edges, depth=0)
        return self

    def _build(
        self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, edges: List[np.ndarray], depth: int
    ) -> _Node:
        n = len(idx)
        ys = y[idx]
        mean_val = float(np.mean(ys)) if n > 0 else 0.0
        if depth >= self.max_depth or n < self.min_samples_split or n < 2 * self.min_samples_leaf:
            return _Node(value=mean_val)

        total_sum = float(np.sum(ys))
        total_sq = float(np.sum(ys * ys))
        parent_imp = total_sq - (total_sum * total_sum) / n
        if parent_imp <= 1e-12:
            return _Node(value=mean_val)

        d = X.shape[1]
        if self.max_features is not None and self.max_features < d:
            feats = self.rng.choice(d, size=self.max_features, replace=False)
        else:
            feats = range(d)

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        for j in feats:
            th = edges[j]
            if len(th) == 0:
                continue
            col = X[idx, j]
            bins = np.searchsorted(th, col, side="right")
            nb = len(th) + 1
            cnts = np.bincount(bins, minlength=nb)
            sums = np.bincount(bins, weights=ys, minlength=nb)
            l_cnt = np.cumsum(cnts[:-1])
            l_sum = np.cumsum(sums[:-1])
            r_cnt = n - l_cnt
            r_sum = total_sum - l_sum

            valid = (l_cnt >= self.min_samples_leaf) & (r_cnt >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            # Gain proportional to l_sum^2 / l_cnt + r_sum^2 / r_cnt - total_sum^2 / n
            score = np.full(len(th), -np.inf, dtype=np.float64)
            score[valid] = (
                (l_sum[valid] ** 2) / l_cnt[valid]
                + (r_sum[valid] ** 2) / r_cnt[valid]
                - (total_sum * total_sum) / n
            )
            b_idx = int(np.argmax(score))
            if score[b_idx] > best_gain:
                best_gain = float(score[b_idx])
                best_feat = int(j)
                best_thresh = float(th[b_idx])

        if best_feat == -1 or best_gain <= 1e-12:
            return _Node(value=mean_val)

        assert self.feature_importances_ is not None
        self.feature_importances_[best_feat] += best_gain
        mask = X[idx, best_feat] <= best_thresh
        left_idx = idx[mask]
        right_idx = idx[~mask]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return _Node(value=mean_val)
        return _Node(
            feature=best_feat,
            threshold=best_thresh,
            left=self._build(X, y, left_idx, edges, depth + 1),
            right=self._build(X, y, right_idx, edges, depth + 1),
            gain=best_gain,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        out = np.empty(X.shape[0], dtype=np.float32)

        def _eval(node: _Node, idx: np.ndarray) -> None:
            if len(idx) == 0:
                return
            if node.feature == -1:
                out[idx] = node.value
                return
            mask = X[idx, node.feature] <= node.threshold
            _eval(node.left, idx[mask])  # type: ignore[arg-type]
            _eval(node.right, idx[~mask])  # type: ignore[arg-type]

        _eval(self.root, np.arange(X.shape[0], dtype=np.int32))
        return out


class _HistTreeClassifier:
    """Fast histogram-binned CART classification tree (Gini impurity)."""

    def __init__(
        self,
        n_classes: int,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        max_bins: int = 32,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_classes = int(n_classes)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_bins = int(max_bins)
        self.rng = rng or np.random.default_rng(42)
        self.root: Optional[_Node] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, bin_edges: Optional[List[np.ndarray]] = None) -> "_HistTreeClassifier":
        n, d = X.shape
        self.feature_importances_ = np.zeros(d, dtype=np.float64)
        edges = bin_edges if bin_edges is not None else _compute_bin_edges(X, self.max_bins)
        idx = np.arange(n, dtype=np.int32)
        self.root = self._build(X, y, idx, edges, depth=0)
        return self

    def _build(
        self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, edges: List[np.ndarray], depth: int
    ) -> _Node:
        n = len(idx)
        ys = y[idx]
        cls_counts = np.bincount(ys, minlength=self.n_classes).astype(np.float64)
        probs = cls_counts / max(n, 1)
        gini_parent = 1.0 - float(np.sum(probs * probs))
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or gini_parent <= 1e-12
        ):
            return _Node(value=probs.astype(np.float32))

        d = X.shape[1]
        if self.max_features is not None and self.max_features < d:
            feats = self.rng.choice(d, size=self.max_features, replace=False)
        else:
            feats = range(d)

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        for j in feats:
            th = edges[j]
            if len(th) == 0:
                continue
            col = X[idx, j]
            bins = np.searchsorted(th, col, side="right")
            nb = len(th) + 1
            # Histogram (nb, n_classes)
            hist = np.zeros((nb, self.n_classes), dtype=np.float64)
            np.add.at(hist, (bins, ys), 1.0)
            l_cls = np.cumsum(hist[:-1], axis=0)
            l_cnt = np.sum(l_cls, axis=1)
            r_cls = cls_counts[None, :] - l_cls
            r_cnt = n - l_cnt

            valid = (l_cnt >= self.min_samples_leaf) & (r_cnt >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            lv = l_cnt[valid][:, None]
            rv = r_cnt[valid][:, None]
            g_l = 1.0 - np.sum((l_cls[valid] / lv) ** 2, axis=1)
            g_r = 1.0 - np.sum((r_cls[valid] / rv) ** 2, axis=1)
            weighted_g = (l_cnt[valid] * g_l + r_cnt[valid] * g_r) / n
            gains = n * (gini_parent - weighted_g)
            score = np.full(len(th), -np.inf, dtype=np.float64)
            score[valid] = gains
            b_idx = int(np.argmax(score))
            if score[b_idx] > best_gain:
                best_gain = float(score[b_idx])
                best_feat = int(j)
                best_thresh = float(th[b_idx])

        if best_feat == -1 or best_gain <= 1e-12:
            return _Node(value=probs.astype(np.float32))

        assert self.feature_importances_ is not None
        self.feature_importances_[best_feat] += best_gain
        mask = X[idx, best_feat] <= best_thresh
        left_idx = idx[mask]
        right_idx = idx[~mask]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return _Node(value=probs.astype(np.float32))
        return _Node(
            feature=best_feat,
            threshold=best_thresh,
            left=self._build(X, y, left_idx, edges, depth + 1),
            right=self._build(X, y, right_idx, edges, depth + 1),
            gain=best_gain,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        out = np.empty((X.shape[0], self.n_classes), dtype=np.float32)

        def _eval(node: _Node, idx: np.ndarray) -> None:
            if len(idx) == 0:
                return
            if node.feature == -1:
                out[idx] = node.value
                return
            mask = X[idx, node.feature] <= node.threshold
            _eval(node.left, idx[mask])  # type: ignore[arg-type]
            _eval(node.right, idx[~mask])  # type: ignore[arg-type]

        _eval(self.root, np.arange(X.shape[0], dtype=np.int32))
        return out


def _resolve_max_features(max_features: Any, d: int) -> int:
    if max_features is None or max_features == "all":
        return d
    if max_features == "sqrt":
        return max(1, int(np.sqrt(d)))
    if max_features == "log2":
        return max(1, int(np.log2(d)))
    if isinstance(max_features, float):
        return max(1, min(d, int(round(max_features * d))))
    return max(1, min(d, int(max_features)))


class RandomForestRegressor:
    """Histogram-binned Random Forest Regressor returning Vulkan GPU arrays."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = 1.0,
        max_bins: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.estimators_: List[_HistTreeRegressor] = []
        self.feature_importances_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any) -> "RandomForestRegressor":
        X_np = to_gpu_2d(X).to_numpy()
        y_np = to_gpu_1d(y).to_numpy().astype(np.float32)
        n, d = X_np.shape
        mf = _resolve_max_features(self.max_features, d)
        edges = _compute_bin_edges(X_np, self.max_bins)
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        imp = np.zeros(d, dtype=np.float64)
        for _ in range(self.n_estimators):
            boot = rng.integers(0, n, size=n)
            tree = _HistTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                max_bins=self.max_bins,
                rng=rng,
            ).fit(X_np[boot], y_np[boot], bin_edges=edges)
            self.estimators_.append(tree)
            if tree.feature_importances_ is not None:
                imp += tree.feature_importances_
        total_imp = float(np.sum(imp))
        self.feature_importances_ = vnp.array(
            (imp / total_imp if total_imp > 0 else imp).astype(np.float32)
        )
        return self

    def predict(self, X: Any) -> vnp.ndarray:
        if not self.estimators_:
            raise RuntimeError("RandomForestRegressor must be fitted before predict()")
        X_np = to_gpu_2d(X).to_numpy()
        acc = np.zeros(X_np.shape[0], dtype=np.float64)
        for tree in self.estimators_:
            acc += tree.predict(X_np)
        return vnp.array((acc / len(self.estimators_)).astype(np.float32))

    def score(self, X: Any, y: Any) -> float:
        return r2_score(y, self.predict(X))


class RandomForestClassifier:
    """Histogram-binned Random Forest Classifier returning Vulkan GPU arrays."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "sqrt",
        max_bins: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.estimators_: List[_HistTreeClassifier] = []
        self.feature_importances_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any) -> "RandomForestClassifier":
        X_np = to_gpu_2d(X).to_numpy()
        y_raw = to_gpu_1d(y).to_numpy().astype(np.int32)
        classes, y_codes = np.unique(y_raw, return_inverse=True)
        self.classes_ = classes
        n, d = X_np.shape
        mf = _resolve_max_features(self.max_features, d)
        edges = _compute_bin_edges(X_np, self.max_bins)
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        imp = np.zeros(d, dtype=np.float64)
        for _ in range(self.n_estimators):
            boot = rng.integers(0, n, size=n)
            tree = _HistTreeClassifier(
                n_classes=len(classes),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                max_bins=self.max_bins,
                rng=rng,
            ).fit(X_np[boot], y_codes[boot], bin_edges=edges)
            self.estimators_.append(tree)
            if tree.feature_importances_ is not None:
                imp += tree.feature_importances_
        total_imp = float(np.sum(imp))
        self.feature_importances_ = vnp.array(
            (imp / total_imp if total_imp > 0 else imp).astype(np.float32)
        )
        return self

    def predict_proba(self, X: Any) -> vnp.ndarray:
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("RandomForestClassifier must be fitted before predict_proba()")
        X_np = to_gpu_2d(X).to_numpy()
        acc = np.zeros((X_np.shape[0], len(self.classes_)), dtype=np.float64)
        for tree in self.estimators_:
            acc += tree.predict_proba(X_np)
        return vnp.array((acc / len(self.estimators_)).astype(np.float32))

    def predict(self, X: Any) -> vnp.ndarray:
        assert self.classes_ is not None
        probs = self.predict_proba(X).to_numpy()
        return vnp.array(self.classes_[np.argmax(probs, axis=1)].astype(np.int32), dtype=np.int32)

    def score(self, X: Any, y: Any) -> float:
        return accuracy_score(y, self.predict(X))


class GradientBoostingRegressor:
    """Histogram-binned Gradient Boosted Decision Trees for Regression (XGBoost/cuML style)."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 2,
        subsample: float = 1.0,
        max_bins: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.init_pred_: float = 0.0
        self.estimators_: List[_HistTreeRegressor] = []
        self.feature_importances_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any) -> "GradientBoostingRegressor":
        X_np = to_gpu_2d(X).to_numpy()
        y_np = to_gpu_1d(y).to_numpy().astype(np.float64)
        n, d = X_np.shape
        edges = _compute_bin_edges(X_np, self.max_bins)
        rng = np.random.default_rng(self.random_state)

        self.init_pred_ = float(np.mean(y_np))
        pred = np.full(n, self.init_pred_, dtype=np.float64)
        self.estimators_ = []
        imp = np.zeros(d, dtype=np.float64)

        sub_n = max(2, int(round(n * self.subsample)))
        for _ in range(self.n_estimators):
            residuals = (y_np - pred).astype(np.float32)
            if self.subsample < 1.0:
                idx = rng.choice(n, size=sub_n, replace=False)
                tree = _HistTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_bins=self.max_bins,
                    rng=rng,
                ).fit(X_np[idx], residuals[idx], bin_edges=edges)
            else:
                tree = _HistTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_bins=self.max_bins,
                    rng=rng,
                ).fit(X_np, residuals, bin_edges=edges)
            pred += self.learning_rate * tree.predict(X_np)
            self.estimators_.append(tree)
            if tree.feature_importances_ is not None:
                imp += tree.feature_importances_

        total_imp = float(np.sum(imp))
        self.feature_importances_ = vnp.array(
            (imp / total_imp if total_imp > 0 else imp).astype(np.float32)
        )
        return self

    def predict(self, X: Any) -> vnp.ndarray:
        if not self.estimators_:
            raise RuntimeError("GradientBoostingRegressor must be fitted before predict()")
        X_np = to_gpu_2d(X).to_numpy()
        pred = np.full(X_np.shape[0], self.init_pred_, dtype=np.float64)
        for tree in self.estimators_:
            pred += self.learning_rate * tree.predict(X_np)
        return vnp.array(pred.astype(np.float32))

    def score(self, X: Any, y: Any) -> float:
        return r2_score(y, self.predict(X))


class GradientBoostingClassifier:
    """Histogram-binned Gradient Boosted Decision Trees for Binary & Multiclass Classification."""

    def __init__(
        self,
        n_estimators: int = 80,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 2,
        subsample: float = 1.0,
        max_bins: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.init_logits_: Optional[np.ndarray] = None
        self.estimators_: List[List[_HistTreeRegressor]] = []
        self.feature_importances_: Optional[vnp.ndarray] = None

    def fit(self, X: Any, y: Any) -> "GradientBoostingClassifier":
        X_np = to_gpu_2d(X).to_numpy()
        y_raw = to_gpu_1d(y).to_numpy().astype(np.int32)
        classes, y_codes = np.unique(y_raw, return_inverse=True)
        self.classes_ = classes
        k = len(classes)
        n, d = X_np.shape
        edges = _compute_bin_edges(X_np, self.max_bins)
        rng = np.random.default_rng(self.random_state)

        priors = np.bincount(y_codes, minlength=k).astype(np.float64) / max(n, 1)
        self.init_logits_ = np.log(np.clip(priors, 1e-6, 1.0))
        logits = np.broadcast_to(self.init_logits_[None, :], (n, k)).copy()
        y_oh = np.eye(k, dtype=np.float64)[y_codes]

        self.estimators_ = []
        imp = np.zeros(d, dtype=np.float64)
        sub_n = max(2, int(round(n * self.subsample)))

        for _ in range(self.n_estimators):
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            step_trees: List[_HistTreeRegressor] = []
            idx = rng.choice(n, size=sub_n, replace=False) if self.subsample < 1.0 else None
            for c in range(k):
                res = (y_oh[:, c] - probs[:, c]).astype(np.float32)
                if idx is not None:
                    tree = _HistTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        max_bins=self.max_bins,
                        rng=rng,
                    ).fit(X_np[idx], res[idx], bin_edges=edges)
                else:
                    tree = _HistTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        max_bins=self.max_bins,
                        rng=rng,
                    ).fit(X_np, res, bin_edges=edges)
                logits[:, c] += self.learning_rate * tree.predict(X_np)
                step_trees.append(tree)
                if tree.feature_importances_ is not None:
                    imp += tree.feature_importances_
            self.estimators_.append(step_trees)

        total_imp = float(np.sum(imp))
        self.feature_importances_ = vnp.array(
            (imp / total_imp if total_imp > 0 else imp).astype(np.float32)
        )
        return self

    def predict_proba(self, X: Any) -> vnp.ndarray:
        if not self.estimators_ or self.classes_ is None or self.init_logits_ is None:
            raise RuntimeError("GradientBoostingClassifier must be fitted before predict_proba()")
        X_np = to_gpu_2d(X).to_numpy()
        k = len(self.classes_)
        logits = np.broadcast_to(self.init_logits_[None, :], (X_np.shape[0], k)).copy()
        for step_trees in self.estimators_:
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


from .xgb import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    XGBClassifier,
    XGBRegressor,
)

__all__ = [
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "XGBClassifier",
    "XGBRegressor",
]
