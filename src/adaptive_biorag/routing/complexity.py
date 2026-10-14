"""Transparent, training-free complexity features + score.

complexity_score = min(1, sum_i w_i * f_i). It is an UNCALIBRATED heuristic index in [0,1], not a probability.
All weights live in config (routing.weights); they are heuristics, not learned or validated values.
"""
from __future__ import annotations

import re

from ..utils.text import entity_like_tokens

_CLAUSE_SEP = re.compile(r";|,\s*(?:and|or|but|which|while|whereas|whether)\b|\band\s+(?:what|which|how|why|whether|who)\b", re.I)
_WH = re.compile(r"\b(what|which|who|whom|how|why|when|where|whether)\b", re.I)
_COMPARE = re.compile(r"\b(compare[sd]?|comparison|versus|vs\.?|differ(?:s|ence|ences)?|better than|worse than|more than|less than)\b", re.I)
_RELATION = re.compile(r"\b(relationship|associat\w*|link(?:ed|s)?|connect\w*|mechanism|interact\w*|pathway|mediat\w*|"
                       r"lead(?:s|ing)? to|result(?:s|ing)? in|affect\w*|role of|due to|cause[sd]?)\b", re.I)
_CHAIN = re.compile(r"\s(?:encoded by|associated with|caused by|mediated by|targeted by|regulated by|expressed in|"
                    r"responsible for|known to|that|which|whose|who)\s", re.I)


def extract_features(question: str) -> dict:
    q = " " + question.strip() + " "
    n_words = len(question.split())
    ents = {e.lower() for e in entity_like_tokens(question)}
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)", question, re.I)
    return {
        "n_words": n_words,
        "n_clauses": 1 + len(_CLAUSE_SEP.findall(question)),
        "n_extra_wh": max(0, len(_WH.findall(question)) - 1),
        "n_question_marks": question.count("?"),
        "comparison_cues": len(_COMPARE.findall(question)),
        "relation_cues": len(_RELATION.findall(question)),
        "n_entities": len(ents) + (1 if between else 0),
        "chain_phrases": len(_CHAIN.findall(q)),
    }


def complexity_score(f: dict, w: dict) -> tuple[float, list[str]]:
    s, reasons = 0.0, []

    def add(code, val):
        nonlocal s
        if val > 0:
            s += val
            reasons.append(code)

    add("multiple_clauses", w["extra_clause"] * min(2, f["n_clauses"] - 1))
    add("multiple_wh_words", w["extra_wh"] * min(2, f["n_extra_wh"]))
    add("comparison_language", w["comparison"] if f["comparison_cues"] else 0)
    add("relationship_language", w["relation"] if f["relation_cues"] else 0)
    add("multiple_entities", w["multi_entity"] if f["n_entities"] >= 2 else 0)
    add("chained_reference", w["chain"] * min(2, f["chain_phrases"]))
    add("long_question", w["long_question"] if f["n_words"] >= 25 else 0)
    add("multiple_question_marks", w["multi_question_mark"] if f["n_question_marks"] > 1 else 0)
    return min(1.0, round(s, 4)), reasons
