"""Exact (normalised-text) passage deduplication that keeps provenance of every dropped duplicate as an alias."""
from __future__ import annotations

from ..schemas import Passage
from ..utils.text import text_hash


def deduplicate_passages(passages: list[Passage]) -> tuple[list[Passage], dict]:
    canon: dict[str, Passage] = {}
    kept: list[Passage] = []
    alias_map: dict[str, str] = {}
    for p in passages:
        h = text_hash(p.text)
        if h in canon:
            c = canon[h]
            c.metadata.setdefault("aliases", []).append({"passage_id": p.passage_id, "document_id": p.document_id})
            alias_map[p.passage_id] = c.passage_id
        else:
            canon[h] = p
            kept.append(p)
    return kept, {"n_input": len(passages), "n_kept": len(kept), "n_duplicates_removed": len(alias_map), "alias_map": alias_map}
