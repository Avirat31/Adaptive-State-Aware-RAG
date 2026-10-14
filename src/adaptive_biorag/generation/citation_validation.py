"""Citation VALIDITY checks: is every cited passage ID one that was actually supplied to the generator?
This is a provenance check, not proof that the cited text entails the claim."""
from __future__ import annotations

import re

from ..schemas import RetrievalResult
from .base import GenerationOutput

_BRACKET = re.compile(r"\[([^\[\]]+)\]")


def extract_inline_citations(text: str) -> list[str]:
    ids = []
    for m in _BRACKET.finditer(text or ""):
        for tok in re.split(r"[,;]", m.group(1)):
            tok = tok.strip()
            if tok and " " not in tok:
                ids.append(tok)
    return ids


def validate_citations(out: GenerationOutput, evidence: list[RetrievalResult]) -> GenerationOutput:
    provided = {r.passage_id: r.document_id for r in evidence}
    invalid: list[str] = []
    for c in out.citations:
        pid = c.get("passage_id")
        if pid not in provided:
            c["valid"] = False
            invalid.append(str(pid))
        elif c.get("document_id") not in (None, "", provided[pid]):
            c["valid"] = False
            c["reason"] = "document_id does not match the supplied passage"
            invalid.append(str(pid))
        else:
            c["document_id"], c["valid"] = provided[pid], True
    have = {c.get("passage_id") for c in out.citations}
    for pid in dict.fromkeys(extract_inline_citations(out.answer)):
        if pid in provided and pid not in have:
            out.citations.append({"passage_id": pid, "document_id": provided[pid], "claim_supported": "", "valid": True})
        elif pid not in provided and pid not in invalid:
            invalid.append(pid)
    out.invalid_citations = invalid
    if invalid:
        out.limitations.append(f"{len(invalid)} citation(s) referenced passages that were not supplied and were marked invalid.")
    return out
