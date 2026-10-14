"""Common dataset interface, validation and statistics. Adapters never fabricate gold answers/evidence."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from ..schemas import Document, Example
from ..utils.paths import resolve_path
from ..utils.serialization import file_sha256


class DatasetError(ValueError):
    pass


class DatasetAdapter(ABC):
    name = "base"
    notes: list[str] = []

    def __init__(self, cfg: dict):
        self.cfg, self.ds = cfg, cfg["dataset"]
        self.info: dict = {"dataset": self.name, "split": self.ds["split"], "files": [], "preprocessing": [], "notes": list(self.notes)}

    def _file(self, default_name: str | None = None) -> Path:
        p = self.ds["path"] or (Path(self.ds["data_dir"]) / default_name if default_name else None)
        if p is None:
            raise DatasetError(f"{self.name}: set dataset.path")
        p = resolve_path(p)
        if not p.exists():
            raise DatasetError(f"{self.name}: file not found: {p}. See data/README.md for the required files and where to get them.")
        self.info["files"].append({"path": str(p), "sha256": file_sha256(p)})
        return p

    @abstractmethod
    def _load_all(self) -> list[Example]: ...

    def load(self) -> list[Example]:
        ex = self._load_all()
        if self.ds["limit"]:
            ex = ex[: self.ds["limit"]]
            self.info["preprocessing"].append(f"limited to first {self.ds['limit']} examples (file order)")
        self.info["n_examples"] = len(ex)
        return ex

    def corpus_documents(self) -> list[Document]:
        raise DatasetError(f"{self.name} does not provide a source corpus; supply corpus.source=jsonl with your own documents.")


def validate_examples(examples: list[Example], strict: bool = True) -> dict:
    errs, ids, qtexts = [], Counter(), Counter()
    for e in examples:
        if not e.question_id or not isinstance(e.question, str) or not e.question.strip():
            errs.append(f"example {e.question_id!r}: empty id or question")
        if not isinstance(e.gold_document_ids, list) or not isinstance(e.gold_passage_ids, list):
            errs.append(f"example {e.question_id!r}: gold id fields must be lists")
        ids[e.question_id] += 1
        qtexts[" ".join(e.question.lower().split())] += 1
    dup_ids = [k for k, v in ids.items() if v > 1]
    if dup_ids:
        errs.append(f"duplicate question_ids: {dup_ids[:5]}")
    rep = {"n_examples": len(examples), "errors": errs, "duplicate_question_ids": dup_ids,
           "duplicate_question_texts": sum(1 for v in qtexts.values() if v > 1),
           "split_counts": dict(Counter(e.split for e in examples)),
           "question_type_counts": dict(Counter(e.question_type for e in examples)),
           "missing": {"gold_answer": sum(e.gold_answer is None for e in examples),
                       "gold_document_ids": sum(not e.gold_document_ids for e in examples),
                       "gold_passage_ids": sum(not e.gold_passage_ids for e in examples)},
           "n_with_gold_evidence": sum(e.has_gold_evidence for e in examples)}
    rep["evidence_retrieval_eval_available"] = rep["n_with_gold_evidence"] > 0
    if errs and strict:
        raise DatasetError("Dataset validation failed: " + " | ".join(errs[:5]))
    return rep
