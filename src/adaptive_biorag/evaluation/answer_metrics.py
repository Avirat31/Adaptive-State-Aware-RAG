"""Answer metrics. Task type is inferred per example:
 - classification: gold in {yes,no,maybe} or gold is an option letter with options present -> accuracy / macro-F1.
   A missing predicted label (generator abstained / extractive baseline) counts as INCORRECT and is reported as `abstained`.
 - text: gold string or list (possibly nested synonym lists, e.g. BioASQ factoid/list) -> EM, 'contains', token-F1, ROUGE-L,
   each = max over all flattened references. NOTE: official BioASQ list/factoid metrics (MRR, list F1) are NOT replicated.
 - examples with gold_answer None are skipped (n_no_gold).
Normalisation: lowercase, strip punctuation and English articles, collapse whitespace.
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ART = re.compile(r"\b(a|an|the)\b")
_CITE = re.compile(r"\[[^\[\]]*\]")


def normalize_answer(s: str) -> str:
    s = _CITE.sub(" ", str(s).lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"-"}).replace("-", " ")
    return " ".join(_ART.sub(" ", s).split())


def flatten_gold(g) -> list[str]:
    if g is None:
        return []
    if isinstance(g, (list, tuple)):
        return [x for e in g for x in flatten_gold(e)]
    return [str(g)]


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))


def contains_match(pred: str, golds: list[str]) -> float:
    p = f" {normalize_answer(pred)} "
    return float(any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g)))


def token_prf(pred: str, gold: str) -> tuple[float, float, float]:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0, 0.0, 0.0
    pr, rc = common / len(p), common / len(g)
    return pr, rc, 2 * pr * rc / (pr + rc)


def rouge_l_f(pred: str, gold: str) -> float:
    a, b = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    l = prev[-1]
    if l == 0:
        return 0.0
    pr, rc = l / len(a), l / len(b)
    return 2 * pr * rc / (pr + rc)


def _label_from(pred: dict, kind: str) -> str | None:
    lab = pred["answer"].get("label")
    if lab:
        lab = str(lab).strip()
        return lab.upper() if kind == "mcq" else lab.lower()
    if kind == "yesno":
        m = re.match(r"\s*(yes|no|maybe)\b", pred["answer"].get("answer", ""), re.I)
        return m.group(1).lower() if m else None
    return None


def score_prediction(pred: dict) -> dict:
    gold, opts = pred["gold"].get("gold_answer"), pred["gold"].get("options")
    if gold is None:
        return {"kind": "none"}
    if isinstance(gold, str) and gold.strip().lower() in {"yes", "no", "maybe"}:
        kind, g = "yesno", gold.strip().lower()
    elif isinstance(gold, str) and isinstance(opts, dict) and gold.strip().upper() in {str(k).upper() for k in opts}:
        kind, g = "mcq", gold.strip().upper()
    else:
        golds = flatten_gold(gold)
        if not golds:
            return {"kind": "none"}
        txt = pred["answer"].get("answer", "")
        return {"kind": "text", "em": exact_match(txt, golds), "contains": contains_match(txt, golds),
                "f1": max(token_prf(txt, g)[2] for g in golds), "precision": max(token_prf(txt, g)[0] for g in golds),
                "recall": max(token_prf(txt, g)[1] for g in golds), "rougeL": max(rouge_l_f(txt, g) for g in golds)}
    p = _label_from(pred, kind)
    return {"kind": "classification", "subtype": kind, "gold": g, "pred": p, "correct": float(p == g), "abstained": p is None}


def macro_f1(golds: list[str], preds: list[str | None]) -> float:
    labels = sorted({*golds, *[p for p in preds if p]})
    f1s = []
    for l in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == l and p == l)
        fp = sum(1 for g, p in zip(golds, preds) if g != l and p == l)
        fn = sum(1 for g, p in zip(golds, preds) if g == l and p != l)
        pr, rc = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate_answers(preds: list[dict]) -> dict:
    sc = [score_prediction(p) for p in preds]
    text = [s for s in sc if s["kind"] == "text"]
    cls = [s for s in sc if s["kind"] == "classification"]
    out: dict = {"n_total": len(preds), "n_no_gold": sum(s["kind"] == "none" for s in sc), "n_text": len(text), "n_classification": len(cls)}
    if text:
        for k in ("em", "contains", "f1", "precision", "recall", "rougeL"):
            out[f"text_{k}"] = sum(s[k] for s in text) / len(text)
    if cls:
        out["cls_accuracy"] = sum(s["correct"] for s in cls) / len(cls)
        out["cls_macro_f1"] = macro_f1([s["gold"] for s in cls], [s["pred"] for s in cls])
        out["cls_abstained"] = sum(s["abstained"] for s in cls)
        out["cls_gold_distribution"] = dict(Counter(s["gold"] for s in cls))
    return out
