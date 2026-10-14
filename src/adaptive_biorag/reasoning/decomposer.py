"""Query decomposition. The deterministic decomposer is a conservative BASELINE and will not match LLM quality."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..logging_utils import get_logger
from ..utils.text import STOPWORDS, entity_like_tokens, tokenize
from .prompts import DECOMPOSE_SYSTEM, decompose_user

log = get_logger("decomposer")

_SPLIT = re.compile(r"\?\s+(?=\S)|;\s*|,\s*(?:and|but|while|whereas|which)\s+(?=\w)|\s+and\s+(?=(?:what|which|how|why|whether|who|where|when)\b)",
                    re.I)
_CHAIN = re.compile(r"\s+(?=(?:encoded by|associated with|caused by|mediated by|targeted by|regulated by|expressed in|responsible for)\b)", re.I)
_BETWEEN = re.compile(r"\b(?:between|of)\s+(.+?)\s+(?:and|versus|vs\.?)\s+(.+?)[?.]?$", re.I)


def _ntoks(s: str) -> int:
    return len(tokenize(s))


def bridge_terms(texts_titles: list[tuple[str, str]], exclude: set[str], max_terms: int = 6) -> list[str]:
    """Entity-like tokens (gene/drug names) from evidence text, then title words; skips words already in the query."""
    out: list[str] = []
    for title, text in texts_titles:
        for e in entity_like_tokens(f"{title} {text}"):
            if e.lower() not in exclude and e not in out:
                out.append(e)
    for title, _ in texts_titles:
        for t in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", title):
            if t.lower() not in STOPWORDS and t.lower() not in exclude and t not in out:
                out.append(t)
    return out[:max_terms]


class QueryDecomposer(ABC):
    @abstractmethod
    def decompose(self, question: str, evidence_state=None) -> list[str]: ...

    def follow_up_query(self, sub_question: str, question: str, bridge_source: list[tuple[str, str]]) -> str:
        """Reformulate an unresolved sub-question using bridge terms from evidence (title, text) pairs."""
        exclude = {t for t in re.findall(r"[a-z0-9\-]+", sub_question.lower())}
        terms = bridge_terms(bridge_source, exclude)
        return f"{sub_question} {' '.join(terms)}".strip() if terms else f"{question} {sub_question}".strip()


class DeterministicDecomposer(QueryDecomposer):
    def __init__(self, max_sub_questions: int = 4):
        self.max_n = max_sub_questions

    def decompose(self, question: str, evidence_state=None) -> list[str]:
        q = " ".join(question.split())
        parts = [p.strip(" ,;") for p in _SPLIT.split(q) if p and p.strip(" ,;")]
        if len(parts) == 1:
            frag = [f.strip() for f in _CHAIN.split(parts[0]) if f.strip()]
            merged: list[str] = []
            for f in frag:                                # merge fragments with < 3 content tokens into the previous one
                if merged and _ntoks(f) < 3:
                    merged[-1] += " " + f
                else:
                    merged.append(f)
            if len(merged) > 1 and _ntoks(merged[-1]) >= 2:
                parts = merged
            else:
                m = _BETWEEN.search(q)
                if m and re.search(r"\b(compare|difference|differ|relationship|association|versus|vs)\b", q, re.I):
                    parts = [m.group(1).strip(), m.group(2).strip()]
        parts = [p if p.endswith("?") else p + "?" for p in parts] if len(parts) > 1 else parts
        seen, out = set(), []
        for p in parts:
            k = " ".join(tokenize(p))
            if k and k not in seen:
                seen.add(k)
                out.append(p)
        return (out or [q])[: self.max_n]


class LLMDecomposer(QueryDecomposer):
    """LLM sub-question generation with validation; falls back to the deterministic decomposer on ANY failure."""

    def __init__(self, client, max_sub_questions: int = 4):
        self.client, self.max_n = client, max_sub_questions
        self.fallback = DeterministicDecomposer(max_sub_questions)
        self.last_info: dict = {}

    def decompose(self, question: str, evidence_state=None) -> list[str]:
        from ..generation.llm_provider import LLMError, parse_json_object
        covered = list(evidence_state.state.covered_sub_questions) if evidence_state is not None else []
        titles = [it.result.title for it in evidence_state.items.values()][:8] if evidence_state is not None else []
        try:
            resp = self.client.complete(DECOMPOSE_SYSTEM, decompose_user(question, self.max_n, covered, titles),
                                        purpose="decompose", json_mode=True)
            raw = parse_json_object(resp.text).get("sub_questions")
            if not isinstance(raw, list):
                raise ValueError("'sub_questions' missing or not a list")
            qtok = set(tokenize(question + " " + " ".join(titles)))
            out, dropped = [], []
            for s in raw:
                if not isinstance(s, str) or not s.strip() or len(s) > 300:
                    dropped.append(str(s)[:60]); continue
                vocab = (question + " " + " ".join(titles)).lower()
                novel = [w for w in re.findall(r"\b[A-Z][A-Za-z0-9\-]+", s)[1:] if w.lower() not in vocab]
                if not (set(tokenize(s)) & qtok) or novel:   # ungrounded, or names an entity absent from question/evidence
                    dropped.append(s[:60]); continue
                if " ".join(tokenize(s)) in {" ".join(tokenize(c)) for c in covered + out}:
                    continue
                out.append(s.strip())
            self.last_info = {"fallback": False, "dropped": dropped, "model": resp.model}
            if not out:
                raise ValueError("no valid sub-questions after validation")
            return out[: self.max_n]
        except (LLMError, ValueError) as e:
            log.warning("LLM decomposition failed (%s); using deterministic fallback", e)
            self.last_info = {"fallback": True, "error": str(e)}
            return self.fallback.decompose(question, evidence_state)


def build_decomposer(cfg: dict, llm_client=None) -> QueryDecomposer:
    n = cfg["decomposition"]["max_sub_questions"]
    if cfg["decomposition"]["method"] == "llm":
        if llm_client is None:
            raise ValueError("decomposition.method=llm requires an LLM client")
        return LLMDecomposer(llm_client, n)
    return DeterministicDecomposer(n)
