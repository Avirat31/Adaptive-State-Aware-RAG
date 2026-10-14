"""Document loading + normalisation. Malformed documents are RECORDED as failures, never silently invented/repaired."""
from __future__ import annotations

import re
from pathlib import Path

from ..schemas import Document
from ..utils.paths import resolve_path
from ..utils.serialization import iter_jsonl

_WS = re.compile(r"[ \t\r\f\v]+")


def normalize_document(raw: dict) -> tuple[Document | None, str | None]:
    """Return (Document, None) or (None, failure_reason)."""
    if not isinstance(raw, dict):
        return None, "record is not an object"
    did = raw.get("document_id", raw.get("id"))
    if did is None or str(did).strip() == "":
        return None, "missing document_id"
    text = raw.get("text", raw.get("abstract"))
    if not isinstance(text, str) or not text.strip():
        return None, "empty or missing text"
    title = raw.get("title") or ""
    return Document(document_id=str(did).strip(), title=_WS.sub(" ", str(title)).strip(),
                    text=_WS.sub(" ", text).replace("\u00a0", " ").strip(), source=str(raw.get("source") or "unknown"),
                    url=raw.get("url"), metadata=dict(raw.get("metadata") or {})), None


def load_documents_jsonl(path: str | Path) -> tuple[list[Document], list[dict]]:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {p}. See data/README.md for how to prepare a corpus.")
    docs, failures, seen = [], [], set()
    for line_no, raw in iter_jsonl(p):
        d, why = normalize_document(raw)
        if d is None:
            failures.append({"line": line_no, "reason": why})
        elif d.document_id in seen:
            failures.append({"line": line_no, "reason": f"duplicate document_id {d.document_id}"})
        else:
            seen.add(d.document_id)
            docs.append(d)
    return docs, failures
