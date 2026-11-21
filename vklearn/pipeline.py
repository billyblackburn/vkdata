"""scikit-learn-compatible Pipeline for chaining GPU transformers & estimators."""

from __future__ import annotations

from typing import Any, List, Tuple


class Pipeline:
    """Chain a sequence of (name, transformer) steps ending with a final estimator."""

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        self.steps = list(steps)
        self.named_steps = dict(steps)

    def fit(self, X: Any, y: Any = None) -> "Pipeline":
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y)
        self.steps[-1][1].fit(Xt, y)
        return self

    def transform(self, X: Any) -> Any:
        Xt = X
        for _, step in self.steps:
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X: Any) -> Any:
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1][1].predict(Xt)

    def score(self, X: Any, y: Any) -> float:
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1][1].score(Xt, y)
