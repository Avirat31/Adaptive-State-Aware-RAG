from __future__ import annotations

from ..schemas import Document, Example
from .base import DatasetAdapter, DatasetError
from .jsonl_adapter import read_json_or_jsonl


class BioASQAdapter(DatasetAdapter):
    """Local BioASQ Task-b style JSON: {"questions": [{"id","body","type","documents":[pubmed urls],
    "snippets":[{"document","text",...}], "exact_answer", "ideal_answer"}]}   (VERIFY against the release you download).

    Data requires registration at participants-area.bioasq.org; nothing is downloaded here. The official files do NOT
    contain PubMed abstracts: for real retrieval evaluation supply a PubMed-derived corpus (corpus.source=jsonl) whose
    document_id = PMID. Gold documents are PMIDs parsed from the `documents` URLs; passage-level gold is unavailable
    (snippets are kept in metadata, offsets are not aligned to our passages), so only document-level retrieval metrics
    are reported. Test-set files may omit answers/evidence -> those fields stay empty (never fabricated).
    """
    name = "bioasq"
    notes = ["Document-level gold only.", "Corpus must be supplied separately unless using the snippet pilot corpus."]

    def _questions(self):
        data = read_json_or_jsonl(self._file())
        qs = data.get("questions") if isinstance(data, dict) else data
        if not isinstance(qs, list):
            raise DatasetError("BioASQ file must contain a 'questions' list")
        return qs

    @staticmethod
    def _pmid(u: str) -> str:
        return str(u).rstrip("/").rsplit("/", 1)[-1]

    def _load_all(self):
        out = []
        for q in self._questions():
            for k in ("id", "body"):
                if k not in q:
                    raise DatasetError(f"BioASQ question lacks '{k}': keys={sorted(q)}")
            gold_ans = q.get("exact_answer", None)
            if gold_ans in (None, [], ""):
                gold_ans = q.get("ideal_answer")
            docs = [self._pmid(u) for u in q.get("documents", []) or []]
            snippets = [{"document": self._pmid(s.get("document", "")), "text": s.get("text", "")} for s in q.get("snippets", []) or []]
            out.append(Example(str(q["id"]), q["body"], "bioasq", self.ds["split"], q.get("type"), gold_ans, docs, [],
                               {"ideal_answer": q.get("ideal_answer"), "gold_snippets": snippets,
                                "no_gold_annotations": not docs and not gold_ans}))
        return out

    def corpus_documents(self):
        by: dict[str, list[str]] = {}
        for q in self._questions():
            for s in q.get("snippets", []) or []:
                pm, t = self._pmid(s.get("document", "")), (s.get("text") or "").strip()
                if pm and t and t not in by.setdefault(pm, []):
                    by[pm].append(t)
        return [Document(pm, "", " ".join(ts), "bioasq_snippets", f"https://pubmed.ncbi.nlm.nih.gov/{pm}/", {"snippet_only": True})
                for pm, ts in by.items()]
