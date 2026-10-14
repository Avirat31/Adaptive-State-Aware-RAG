"""Retrieval metrics over a RANKED list of passages (best first).

Document-level: a ranked passage 'hits' document d if its document_id (or the document_id of a deduplicated alias of that
passage) equals d. Passage-level metrics are computed ONLY for examples with gold_passage_ids.
All metrics are binary-relevance (no graded labels exist in the supported datasets, so nDCG uses 0/1 gains).
Examples without gold labels are EXCLUDED from retrieval aggregates (counted in n_missing), never scored as failures.
"""
from __future__ import annotations

import math


def _sets(ranking: list[dict], level: str) -> list[set]:
    if level == "doc":
        return [{r["document_id"], *r.get("alias_documents", [])} for r in ranking]
    return [{r["passage_id"], *r.get("alias_passages", [])} for r in ranking]


def recall_at_k(sets: list[set], gold: set, k: int) -> float:
    return len(gold & set().union(*sets[:k])) / len(gold) if sets[:k] else 0.0


def hit_at_k(sets, gold, k) -> float:
    return float(any(s & gold for s in sets[:k]))


def mrr(sets, gold) -> float:
    for i, s in enumerate(sets, 1):
        if s & gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(sets, gold, k) -> float:
    seen, dcg = set(), 0.0
    for i, s in enumerate(sets[:k], 1):
        new = (s & gold) - seen
        if new:
            seen |= {sorted(new)[0]}                   # one gold item credited per ranked passage
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(preds: list[dict], ks: list[int]) -> dict:
    out: dict = {"n_total": len(preds)}
    for level, key in (("doc", "gold_document_ids"), ("passage", "gold_passage_ids")):
        rows = [(p["retrieval_ranking"], set(p["gold"][key])) for p in preds if p["gold"].get(key)]
        out[f"n_with_gold_{level}"] = len(rows)
        out[f"n_missing_gold_{level}"] = len(preds) - len(rows)
        if not rows:
            out[f"{level}_metrics"] = None
            continue
        m: dict = {"mrr": sum(mrr(_sets(r, level), g) for r, g in rows) / len(rows)}
        for k in ks:
            m[f"recall@{k}"] = sum(recall_at_k(_sets(r, level), g, k) for r, g in rows) / len(rows)
            m[f"hit@{k}"] = sum(hit_at_k(_sets(r, level), g, k) for r, g in rows) / len(rows)
            m[f"ndcg@{k}"] = sum(ndcg_at_k(_sets(r, level), g, k) for r, g in rows) / len(rows)
        out[f"{level}_metrics"] = m
    return out
