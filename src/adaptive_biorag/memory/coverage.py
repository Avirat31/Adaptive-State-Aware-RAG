"""Evidence-coverage definition (lexical proxy; documented in README).

support(sq, passage) = |content_tokens(sq) ∩ tokens(title + passage)| / |content_tokens(sq)|
A sub-question is COVERED iff some UNIQUE stored passage has support >= memory.subq_support_threshold.
Coverage = c / m for m > 0 sub-questions. For m == 0: 0.0 if fewer than `min_evidence` unique passages exist, else 1.0.
This is a transparent proxy for "has candidate evidence", NOT a proof that the evidence answers the sub-question.
"""
from __future__ import annotations

from ..utils.text import tokenize


def token_set(text: str) -> frozenset:
    return frozenset(tokenize(text))


def support_score(subq_tokens: frozenset, passage_tokens: frozenset) -> float:
    if not subq_tokens:
        return 0.0
    return len(subq_tokens & passage_tokens) / len(subq_tokens)


def compute_coverage(n_subq: int, n_covered: int, n_unique_evidence: int, min_evidence: int = 1) -> float:
    if n_subq == 0:
        return 1.0 if n_unique_evidence >= min_evidence else 0.0
    return n_covered / n_subq
