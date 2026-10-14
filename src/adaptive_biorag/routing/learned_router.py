"""Optional supervised router (logistic regression on the interpretable heuristic features, pure NumPy).

Only usable when TRUSTWORTHY gold complexity labels exist (e.g. a small human-annotated subset). This repository does
NOT ship labels and does not invent them. Train on the training split only, choose the threshold on validation only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schemas import RouteDecision
from .complexity import extract_features
from .heuristic_router import Router

FEATS = ["n_words", "n_clauses", "n_extra_wh", "n_question_marks", "comparison_cues", "relation_cues", "n_entities", "chain_phrases"]


def _x(question: str) -> np.ndarray:
    f = extract_features(question)
    return np.array([min(f[k], 40) / (40.0 if k == "n_words" else 4.0) for k in FEATS])


class LearnedRouter(Router):
    def __init__(self, w: np.ndarray | None = None, b: float = 0.0, threshold: float = 0.5):
        self.w, self.b, self.threshold = w, b, threshold

    def fit(self, questions: list[str], labels: list[int], epochs: int = 500, lr: float = 0.5, l2: float = 1e-3):
        if len(questions) < 4 or len(set(labels)) < 2:
            raise ValueError("Need >= 4 labelled questions covering both classes (1=complex, 0=simple).")
        X, y = np.stack([_x(q) for q in questions]), np.asarray(labels, float)
        w, b = np.zeros(X.shape[1]), 0.0
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(X @ w + b)))
            w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
            b -= lr * float(np.mean(p - y))
        self.w, self.b = w, b
        return self

    def proba(self, question: str) -> float:
        if self.w is None:
            raise RuntimeError("LearnedRouter is not trained")
        return float(1 / (1 + np.exp(-(_x(question) @ self.w + self.b))))

    def tune_threshold(self, questions: list[str], labels: list[int]) -> float:
        """Pick the threshold maximising macro-F1 on VALIDATION data."""
        best, bt = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            m = classification_report([int(self.proba(q) >= t) for q in questions], labels)["macro_f1"]
            if m > best:
                best, bt = m, float(t)
        self.threshold = bt
        return bt

    def route(self, question: str) -> RouteDecision:
        p = self.proba(question)
        return RouteDecision("multi_step" if p >= self.threshold else "single_step", round(p, 4), ["learned_classifier"],
                             extract_features(question), "learned")

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"w": self.w.tolist(), "b": self.b, "threshold": self.threshold, "feats": FEATS}))

    @classmethod
    def load(cls, path: str | Path, threshold: float | None = None) -> "LearnedRouter":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["w"]), d["b"], d["threshold"] if threshold is None else threshold)


def classification_report(pred: list[int], gold: list[int]) -> dict:
    tp = {c: sum(1 for p, g in zip(pred, gold) if p == c and g == c) for c in (0, 1)}
    fp = {c: sum(1 for p, g in zip(pred, gold) if p == c and g != c) for c in (0, 1)}
    fn = {c: sum(1 for p, g in zip(pred, gold) if p != c and g == c) for c in (0, 1)}
    f1s = []
    for c in (0, 1):
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    n = max(1, len(gold))
    return {"accuracy": sum(1 for p, g in zip(pred, gold) if p == g) / n, "macro_f1": sum(f1s) / 2,
            "confusion_matrix": {"tn": tp[0], "fp": fp[1],
                                 "fn": sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1), "tp": tp[1]},
            "class_balance": {"simple": gold.count(0), "complex": gold.count(1)}}
