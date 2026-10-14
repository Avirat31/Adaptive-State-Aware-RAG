"""Conservative contradiction CANDIDATE detection (lexical polarity heuristic).

Output is always status='unverified'/'possible_conflict'. Opposite-looking sentences may concern different populations,
doses, outcomes or time frames; this module never decides which claim is medically correct.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..utils.text import split_sentences, tokenize

_NEG = re.compile(r"\b(no|not|n't|without|failed to|fail(?:s|ed)? to|lack(?:s|ed)?|absence of|non-significant|"
                  r"no significant|ineffective|did not)\b", re.I)
_POS = re.compile(r"\b(significantly|improved|increased|reduced|lower|higher|effective|associated with|benefit)\b", re.I)


def polarity(sentence: str) -> str | None:
    if _NEG.search(sentence):
        return "negative"
    return "positive" if _POS.search(sentence) else None


class ContradictionDetector(ABC):
    @abstractmethod
    def detect(self, new_item, existing_items: list) -> list[dict]: ...


class LexicalPolarityDetector(ContradictionDetector):
    method = "lexical_polarity_heuristic"

    def __init__(self, min_overlap: float = 0.35, max_pairs: int = 2000):
        self.min_overlap, self.max_pairs = min_overlap, max_pairs

    def detect(self, new_item, existing_items):
        found, pairs = [], 0
        for s1 in split_sentences(new_item.result.text):
            p1, t1 = polarity(s1), set(tokenize(s1))
            if p1 is None or not t1:
                continue
            for other in existing_items:
                if other.result.document_id == new_item.result.document_id:
                    continue
                for s2 in split_sentences(other.result.text):
                    pairs += 1
                    if pairs > self.max_pairs:
                        return found
                    p2, t2 = polarity(s2), set(tokenize(s2))
                    if p2 is None or p2 == p1 or not t2:
                        continue
                    jac = len(t1 & t2) / len(t1 | t2)
                    if jac >= self.min_overlap:
                        found.append({"status": "possible_conflict", "method": self.method, "overlap": round(jac, 3),
                                      "passage_ids": [new_item.result.passage_id, other.result.passage_id],
                                      "document_ids": [new_item.result.document_id, other.result.document_id],
                                      "claims": [s1, s2],
                                      "note": "Unverified: sentences overlap lexically with opposite polarity; they may "
                                              "concern different populations, outcomes or settings."})
        return found
