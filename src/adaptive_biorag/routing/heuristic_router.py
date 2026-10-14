from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import RouteDecision
from .complexity import complexity_score, extract_features


class Router(ABC):
    @abstractmethod
    def route(self, question: str) -> RouteDecision: ...


class HeuristicRouter(Router):
    def __init__(self, threshold: float = 0.4, weights: dict | None = None):
        from ..config import DEFAULTS
        self.threshold, self.weights = threshold, weights or DEFAULTS["routing"]["weights"]

    def route(self, question: str) -> RouteDecision:
        f = extract_features(question)
        score, reasons = complexity_score(f, self.weights)
        return RouteDecision("multi_step" if score >= self.threshold else "single_step", score, reasons or ["no_complexity_cues"],
                             f, "heuristic")


class FixedRouter(Router):
    """Routing disabled: every question takes the same strategy (needed for fair baselines)."""
    def __init__(self, route: str):
        self._route = route

    def route(self, question: str) -> RouteDecision:
        return RouteDecision(self._route, 0.0, ["routing_disabled"], {}, "fixed")


def build_router(cfg: dict) -> Router:
    r = cfg["routing"]
    if not r["enabled"]:
        return FixedRouter(r["forced_route"])
    if r["method"] == "learned":
        from .learned_router import LearnedRouter
        return LearnedRouter.load(r["learned_model_path"], r["complexity_threshold"])
    return HeuristicRouter(r["complexity_threshold"], r["weights"])
