"""Citation checks.
- validity: cited passage ids that were actually supplied to the generator / all citations (provenance check).
- coverage: fraction of answer sentences that carry >=1 VALID inline citation (claims = sentences; crude segmentation).
- lexical_support: mean token-overlap of a sentence's content words with its cited passage(s). This is a cheap PROXY, NOT
  entailment and NOT human judgment; use the inspection sample (inspection_sample.jsonl) for manual review.
Answers that state insufficient evidence contain no claims and are excluded from coverage/support.
"""
from __future__ import annotations

import re

from ..generation.citation_validation import extract_inline_citations
from ..utils.text import split_sentences, tokenize


_LEAD = re.compile(r"^((?:\[[^\[\]]+\]\s*)+)(.*)$", re.S)


def split_claims(text: str) -> list[str]:
    """Sentence split, then re-attach citation markers that follow the full stop ('... nM. [D1::p0] Next ...')."""
    out: list[str] = []
    for seg in split_sentences(text):
        m = _LEAD.match(seg)
        if m and out:
            out[-1] += " " + m.group(1).strip()
            seg = m.group(2).strip()
        if seg:
            out.append(seg)
    return out


def evaluate_citations(preds: list[dict]) -> dict:
    n_cit = n_valid = 0
    cov, sup, n_claim_answers, n_insuff = [], [], 0, 0
    for p in preds:
        a = p["answer"]
        cits = a.get("citations", [])
        n_cit += len(cits)
        n_valid += sum(1 for c in cits if c.get("valid"))
        n_valid_inline_bad = len(a.get("invalid_citations", []))
        if a.get("insufficient_evidence"):
            n_insuff += 1
            continue
        passages = {e["passage_id"]: e for e in p["final_evidence"]}
        sents = split_claims(a.get("answer", ""))
        if not sents:
            continue
        n_claim_answers += 1
        covered, sups = 0, []
        for s in sents:
            ids = [i for i in extract_inline_citations(s) if i in passages]
            if ids:
                covered += 1
                st = set(tokenize(s))
                if st:
                    sups.append(max(len(st & set(tokenize(passages[i]["text"]))) / len(st) for i in ids))
        cov.append(covered / len(sents))
        if sups:
            sup.append(sum(sups) / len(sups))
    return {"n_total": len(preds), "n_citations": n_cit, "citation_validity": (n_valid / n_cit) if n_cit else None,
            "n_answers_with_claims": n_claim_answers, "n_insufficient_evidence_answers": n_insuff,
            "citation_coverage": (sum(cov) / len(cov)) if cov else None,
            "lexical_support_proxy": (sum(sup) / len(sup)) if sup else None}
