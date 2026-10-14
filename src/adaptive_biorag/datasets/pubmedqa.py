from __future__ import annotations

import json

from ..schemas import Document, Example
from ..utils.paths import resolve_path
from .base import DatasetAdapter, DatasetError


class PubMedQAAdapter(DatasetAdapter):
    """Local PubMedQA files. Expected record format (per the public PubMedQA release; VERIFY against your download):
        {pmid: {"QUESTION": str, "CONTEXTS": [str], "LABELS": [...], "MESHES": [...], "final_decision": "yes|no|maybe",
                "LONG_ANSWER": str, ...}}
    `dataset.path` may be the whole expert-labelled file (e.g. ori_pqal.json) or a fold split file with the same record
    layout; or set `dataset.split_file` to a JSON list/dict of PMIDs to select an official split.

    Task variant implemented: RETRIEVAL-BASED QA (the original abstract context is NOT given to the pipeline). The
    context-provided ("reasoning-required") task is a different setting - its CONTEXTS are only exposed in
    example.metadata when dataset.provide_context=true and the pipeline never reads metadata.
    """
    name = "pubmedqa"
    notes = ["Retrieval variant: question only; contexts removed from the model input.",
             "Closed-pool corpus (source=pubmedqa_contexts) contains the question's own abstract -> optimistic retrieval setting."]

    def _records(self) -> dict:
        p = self._file("ori_pqal.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise DatasetError(f"{p}: expected a JSON object mapping PMID -> record")
        sf = self.ds["split_file"]
        if sf:
            sp = resolve_path(sf)
            if not sp.exists():
                raise DatasetError(f"split_file not found: {sp}")
            ids = json.loads(sp.read_text(encoding="utf-8"))
            ids = list(ids.keys()) if isinstance(ids, dict) else list(ids)
            missing = [i for i in ids if i not in data and str(i) not in data]
            if missing:
                raise DatasetError(f"{len(missing)} ids in split_file are absent from {p}: {missing[:5]}")
            data = {str(i): data[str(i)] for i in ids}
            self.info["preprocessing"].append(f"restricted to ids in {sp.name}")
        return data

    def _load_all(self):
        out = []
        for pmid, rec in self._records().items():
            for k in ("QUESTION", "final_decision"):
                if k not in rec:
                    raise DatasetError(f"PubMedQA record {pmid} lacks '{k}'; schema differs from the expected format")
            md = {"meshes": rec.get("MESHES"), "year": rec.get("YEAR"), "long_answer_reference": rec.get("LONG_ANSWER")}
            if self.ds["provide_context"]:
                md["contexts"] = rec.get("CONTEXTS")
            out.append(Example(str(pmid), rec["QUESTION"], "pubmedqa", self.ds["split"], "yesno_maybe",
                               str(rec["final_decision"]).lower(), [str(pmid)], [], md))
        return out

    def corpus_documents(self):
        docs = []
        for pmid, rec in self._records().items():
            ctx = rec.get("CONTEXTS") or []
            if ctx:
                docs.append(Document(str(pmid), "", " ".join(ctx), "pubmedqa", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                     {"closed_pool": True}))
        return docs
