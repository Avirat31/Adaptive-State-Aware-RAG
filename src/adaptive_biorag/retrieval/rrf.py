"""Reciprocal Rank Fusion: score(d) = sum_r 1 / (k0 + rank_r(d)), ranks start at 1. Raw scores are never added."""
from __future__ import annotations

from ..schemas import RetrievalResult


def rrf_fuse(rankings: dict[str, list[RetrievalResult]], k0: int = 60, top_k: int | None = None) -> list[RetrievalResult]:
    if k0 < 1:
        raise ValueError("k0 must be >= 1")
    fused: dict[str, RetrievalResult] = {}
    for name, results in rankings.items():
        for pos, r in enumerate(results, 1):          # rank = position in the list (robust to caller-set ranks)
            f = fused.get(r.passage_id)
            if f is None:
                f = RetrievalResult(r.document_id, r.passage_id, r.text, r.title, r.source, 0.0, "hybrid_rrf", 0,
                                    {"aliases": r.metadata.get("aliases", []), "ranks": {}, "scores": {}})
                fused[r.passage_id] = f
            if name in f.metadata["ranks"]:           # duplicate within one list: keep the best (first) occurrence
                continue
            f.metadata["ranks"][name] = pos
            f.metadata["scores"][name] = r.score
            f.score += 1.0 / (k0 + pos)
    out = sorted(fused.values(), key=lambda x: (-x.score, x.passage_id))
    if top_k is not None:
        out = out[:top_k]
    for i, r in enumerate(out, 1):
        r.rank = i
    return out
