"""Explicit, testable stopping rules for the iterative loop (evaluated in this order after every iteration)."""
from __future__ import annotations

from ..schemas import StopDecision
from .evidence_store import EvidenceStore


class StoppingPolicy:
    def __init__(self, max_iterations: int = 3, minimum_new_evidence: int = 1, coverage_threshold: float = 1.0,
                 max_duplicate_ratio: float = 0.9, min_evidence_passages: int = 1):
        self.max_iterations, self.min_new = max_iterations, minimum_new_evidence
        self.cov_thr, self.max_dup, self.min_ev = coverage_threshold, max_duplicate_ratio, min_evidence_passages

    @classmethod
    def from_config(cls, cfg: dict) -> "StoppingPolicy":
        s = cfg["stopping"]
        return cls(s["max_iterations"], s["minimum_new_evidence"], s["evidence_coverage_threshold"],
                   s["max_duplicate_ratio"], s["min_evidence_passages"])

    def decide(self, store: EvidenceStore, iteration: int, new_unique: int, total_retrieved: int, duplicates: int,
               failed: bool = False) -> StopDecision:
        cov = store.coverage()
        unresolved = list(store.state.unresolved_sub_questions)
        conflict = "+possible_conflicts" if store.state.conflicting_claims else ""
        enough = len(store.items) >= self.min_ev
        if failed:
            return StopDecision(True, "retrieval_failure", cov, unresolved)
        if cov >= self.cov_thr and enough:
            return StopDecision(True, "sufficient_coverage" + conflict, cov, unresolved)
        if iteration >= self.max_iterations:
            return StopDecision(True, "max_iterations_reached" + conflict, cov, unresolved)
        if new_unique < self.min_new:
            return StopDecision(True, "no_new_evidence" + conflict, cov, unresolved)
        if total_retrieved > 0 and duplicates / total_retrieved >= self.max_dup:
            return StopDecision(True, "mostly_duplicates" + conflict, cov, unresolved)
        return StopDecision(False, "continue_unresolved_subquestions", cov, unresolved)
