"""Shared lightweight text utilities (tokenisation, crude stemming, sentence splitting)."""
from __future__ import annotations

import hashlib
import re

STOPWORDS = frozenset(
    "a an the of in on at to for with by from and or is are was were be been being what which who whom whose how why "
    "when where does do did that this these those as it its into than then there their they can could may might "
    "should would will has have had not no vs versus between about also if so such".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def stem(tok: str) -> str:
    """Very light suffix stripper (NOT Porter). Deterministic and dependency-free."""
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 5 and tok.endswith("ing"):
        tok = tok[:-3]
    elif len(tok) > 4 and tok.endswith("ed"):
        tok = tok[:-2]
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        tok = tok[:-1]
    if len(tok) > 4 and tok.endswith("e"):
        tok = tok[:-1]
    return tok


def tokenize(text: str, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    toks = _TOKEN.findall((text or "").lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks] if do_stem else toks


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split((text or "").strip()) if s.strip()]


def entity_like_tokens(text: str) -> list[str]:
    """Tokens that look like gene/drug/acronym names: contain a digit or an uppercase letter after position 0."""
    out = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text or ""):
        if any(c.isdigit() for c in raw) or any(c.isupper() for c in raw[1:]):
            out.append(raw)
    return out
