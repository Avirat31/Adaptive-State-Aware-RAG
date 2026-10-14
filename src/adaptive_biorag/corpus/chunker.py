"""Word-window passage segmentation with stable IDs and character offsets (source boundaries are never crossed)."""
from __future__ import annotations

import re

from ..schemas import Document, Passage

_WORD = re.compile(r"\S+")


def chunk_document(doc: Document, size: int, overlap: int, min_words: int = 3) -> list[Passage]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("require size >= 1 and 0 <= overlap < size")
    words = [(m.start(), m.end()) for m in _WORD.finditer(doc.text)]
    if len(words) < min_words:
        return []
    step = size - overlap
    out: list[Passage] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        s, e = words[start][0], words[end - 1][1]
        out.append(Passage(passage_id=f"{doc.document_id}::p{len(out)}", document_id=doc.document_id,
                           text=doc.text[s:e], title=doc.title, source=doc.source, start_offset=s, end_offset=e,
                           metadata={"url": doc.url, **{k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float))}}))
        if end >= len(words):
            break
        start += step
    return out
