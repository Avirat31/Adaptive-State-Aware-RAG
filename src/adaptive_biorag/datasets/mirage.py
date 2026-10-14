from __future__ import annotations

import json

from ..schemas import Example
from .base import DatasetAdapter, DatasetError


class MirageAdapter(DatasetAdapter):
    """MIRAGE `benchmark.json` (Teddy-XiongGZ/MIRAGE; 7,663 questions over five subsets; ground-truth snippets removed).
    Assumed layout (VERIFY on your copy): {subset: {qid: {"question": str, "options": {"A": str, ...}, "answer": "A"}}}.
    Select a subset with dataset.subset. MIRAGE provides NO corpus and NO gold documents: retrieval metrics are
    unavailable and an external corpus (corpus.source=jsonl) is required. Questions are multiple-choice; most subsets are
    NOT multi-hop. MIRAGE is an evaluation benchmark - there is no official train/dev split, so threshold/prompt tuning
    must use a held-out portion you define and document.
    """
    name = "mirage"
    notes = ["Evaluation-only benchmark; no gold evidence; external corpus required."]

    def _load_all(self):
        p = self._file("benchmark.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise DatasetError("benchmark.json: expected an object keyed by subset name")
        sub = self.ds["subset"]
        if sub not in data:
            raise DatasetError(f"dataset.subset={sub!r} not found; available subsets: {sorted(data)}")
        out = []
        for qid, rec in data[sub].items():
            if "question" not in rec:
                raise DatasetError(f"{sub}/{qid}: no 'question' field (keys={sorted(rec)})")
            opts = rec.get("options")
            out.append(Example(f"{sub}:{qid}", rec["question"], f"mirage/{sub}", self.ds["split"], "multiple_choice",
                               rec.get("answer"), [], [], {"options": opts if isinstance(opts, dict) else None}))
        return out
