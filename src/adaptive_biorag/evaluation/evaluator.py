from __future__ import annotations

from collections import Counter

from .answer_metrics import evaluate_answers
from .citation_metrics import evaluate_citations
from .efficiency_metrics import aggregate_efficiency
from .retrieval_metrics import evaluate_retrieval


def _block(preds: list[dict], ks: list[int]) -> dict:
    b = {"n": len(preds), "retrieval": evaluate_retrieval(preds, ks), "answer": evaluate_answers(preds),
         "citations": evaluate_citations(preds), "efficiency": aggregate_efficiency([p["efficiency"] for p in preds]),
         "routing": {"route_counts": dict(Counter(p["route"] for p in preds)),
                     "mean_complexity_score": (sum(p["routing"]["complexity_score"] for p in preds) / len(preds)) if preds else None,
                     "mean_iterations": (sum(p["n_iterations"] for p in preds) / len(preds)) if preds else None,
                     "stop_reasons": dict(Counter(str(p["stop_reason"]).split("+")[0] for p in preds)),
                     "mean_evidence_coverage_proxy": (sum(p["evidence_coverage"] for p in preds) / len(preds)) if preds else None}}
    return b


def evaluate_predictions(preds: list[dict], ks: list[int] = (1, 5, 10)) -> dict:
    ks = list(ks)
    has = lambda p: bool(p["gold"].get("gold_document_ids") or p["gold"].get("gold_passage_ids"))
    subsets = {
        "single_step_route": [p for p in preds if p["route"] == "single_step"],
        "multi_step_route": [p for p in preds if p["route"] == "multi_step"],
        "with_gold_evidence": [p for p in preds if has(p)],
        "without_gold_evidence": [p for p in preds if not has(p)],
        "extra_retrieval_triggered": [p for p in preds if p["n_iterations"] > 1],
        "insufficient_evidence_answer": [p for p in preds if p["answer"].get("insufficient_evidence")],
    }
    res = {"overall": _block(preds, ks), "subsets": {k: _block(v, ks) for k, v in subsets.items() if v}}
    res["n_errors"] = sum(1 for p in preds if p.get("errors"))
    return res


def flatten_metrics(res: dict) -> list[dict]:
    rows = []

    def walk(prefix, obj, subset):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(prefix + [k], v, subset)
        elif isinstance(obj, (int, float)) or obj is None:
            rows.append({"subset": subset, "metric": ".".join(prefix), "value": obj})

    walk([], res["overall"], "overall")
    for name, blk in res["subsets"].items():
        walk([], blk, name)
    return rows
