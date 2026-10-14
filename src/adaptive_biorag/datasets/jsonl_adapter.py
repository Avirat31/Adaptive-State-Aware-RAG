from __future__ import annotations

import json

from ..schemas import Example
from ..utils.serialization import iter_jsonl
from .base import DatasetAdapter, DatasetError


def read_json_or_jsonl(p):
    if p.suffix == ".jsonl":
        return [o for _, o in iter_jsonl(p)]
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DatasetError(f"{p}: malformed JSON ({e})") from e


class JsonlAdapter(DatasetAdapter):
    """Internal schema JSONL (used by the synthetic example and for any custom dataset)."""
    name = "jsonl"

    def _load_all(self):
        default = "sample_questions.jsonl" if self.cfg["dataset"]["name"] == "synthetic" else None
        rows = read_json_or_jsonl(self._file(default))
        out = []
        for i, d in enumerate(rows):
            try:
                out.append(Example.from_dict(d))
            except KeyError as e:
                raise DatasetError(f"record {i}: missing required field {e}; internal schema needs question_id and question") from e
        return out
