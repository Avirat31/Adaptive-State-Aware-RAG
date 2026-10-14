from __future__ import annotations

from ..schemas import Example
from .base import DatasetAdapter, DatasetError
from .jsonl_adapter import read_json_or_jsonl


class MedHopQAAdapter(DatasetAdapter):
    """MedHopQA generic JSON/JSONL adapter driven by `dataset.field_map`.

    The official MedHopQA schema/evaluation protocol was NOT verified when this adapter was written (the release could not
    be inspected offline). Therefore nothing is assumed beyond a configurable mapping:
        field_map: {id: <field>, question: <field>, answer: <field>, gold_docs: <field>, question_type: <field>}
    Defaults try common names and fail with the list of available keys if the question field cannot be found.
    Use the dataset's own source corpus (corpus.source=jsonl, with its real article IDs); do not substitute PubMed.
    Do NOT compare MedHopQA scores with PubMed-based benchmarks as if the settings were identical.
    """
    name = "medhopqa"
    notes = ["Schema unverified: confirm field_map against the official release before reporting results."]
    _CAND = {"id": ["question_id", "id", "qid"], "question": ["question", "body", "query"], "answer": ["answer", "gold_answer", "label"],
             "gold_docs": ["gold_document_ids", "supporting_docs", "supporting_documents", "gold_docs"],
             "question_type": ["question_type", "type"]}

    def _pick(self, rec: dict, key: str):
        fm = self.ds["field_map"] or {}
        cands = [fm[key]] if key in fm else self._CAND[key]
        for c in cands:
            if c in rec:
                return rec[c]
        return None

    def _load_all(self):
        data = read_json_or_jsonl(self._file())
        if isinstance(data, dict):
            data = data.get("data") or data.get("questions") or list(data.values())
        out = []
        for i, rec in enumerate(data):
            q = self._pick(rec, "question")
            if not isinstance(q, str):
                raise DatasetError(f"record {i}: question field not found; keys={sorted(rec)}. Set dataset.field_map.question")
            docs = self._pick(rec, "gold_docs") or []
            out.append(Example(str(self._pick(rec, "id") or f"medhop_{i}"), q, "medhopqa", self.ds["split"],
                               self._pick(rec, "question_type") or "multihop", self._pick(rec, "answer"),
                               [str(d) for d in (docs if isinstance(docs, list) else [docs])], [], {}))
        return out
