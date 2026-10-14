"""Corpus -> passages -> persisted index directory (passages.jsonl + manifest.json + cached dense embeddings)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..schemas import Document, Passage
from ..utils.paths import resolve_path
from ..utils.serialization import read_json, read_jsonl, write_json, write_jsonl
from .chunker import chunk_document
from .deduplication import deduplicate_passages
from .loader import load_documents_jsonl


def load_corpus_documents(cfg: dict) -> tuple[list[Document], list[dict], dict]:
    """Return (documents, failures, provenance). Never downloads anything."""
    c, ds = cfg["corpus"], cfg["dataset"]
    if c["source"] == "jsonl":
        docs, fails = load_documents_jsonl(c["path"])
        return docs, fails, {"source": "jsonl", "path": str(c["path"])}
    from ..datasets import get_adapter
    adapter = get_adapter(cfg)
    if c["source"] == "pubmedqa_contexts":
        docs = adapter.corpus_documents()
    elif c["source"] == "bioasq_snippets":
        docs = adapter.corpus_documents()
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(c["source"])
    return docs, [], {"source": c["source"], "dataset": ds["name"],
                      "warning": "closed-pool corpus derived from the benchmark itself; see README 'Corpus caveats'"}


def build_passages(docs: list[Document], size: int, overlap: int, min_words: int) -> tuple[list[Passage], dict]:
    allp: list[Passage] = []
    unmapped: list[str] = []
    for d in docs:
        ps = chunk_document(d, size, overlap, min_words)
        if not ps:
            unmapped.append(d.document_id)
        allp.extend(ps)
    kept, rep = deduplicate_passages(allp)
    rep.update({"n_documents": len(docs), "documents_without_passages": unmapped})
    return kept, rep


def corpus_hash(passages: list[Passage]) -> str:
    h = hashlib.sha256()
    for p in passages:
        h.update(p.passage_id.encode())
        h.update(b"\0")
        h.update(p.text.encode())
        h.update(b"\1")
    return h.hexdigest()[:16]


def build_corpus_index(cfg: dict, force: bool = False) -> tuple[list[Passage], dict, Path]:
    """Build (or reuse) the passage store. Reuse only if the manifest matches the current chunking config + inputs."""
    c = cfg["corpus"]
    index_dir = Path(c["index_dir"])
    if not index_dir.is_absolute():
        index_dir = Path.cwd() / index_dir
    docs, failures, prov = load_corpus_documents(cfg)
    passages, rep = build_passages(docs, c["passage_size"], c["passage_overlap"], c["min_passage_words"])
    manifest = {"adaptive_biorag_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
                "corpus_provenance": prov, "chunking": {k: c[k] for k in ("passage_size", "passage_overlap", "min_passage_words")},
                "n_documents": len(docs), "n_passages": len(passages), "corpus_hash": corpus_hash(passages),
                "load_failures": failures, "dedup": {k: v for k, v in rep.items() if k != "alias_map"},
                "n_unmapped_documents": len(rep["documents_without_passages"])}
    mpath = index_dir / "manifest.json"
    if not force and mpath.exists():
        old = read_json(mpath)
        if old.get("corpus_hash") == manifest["corpus_hash"] and old.get("chunking") == manifest["chunking"]:
            return passages, old, index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(index_dir / "passages.jsonl", (p.to_dict() for p in passages))
    write_json(index_dir / "alias_map.json", rep["alias_map"])
    write_json(mpath, manifest)
    return passages, manifest, index_dir


def load_passages(index_dir: str | Path) -> list[Passage]:
    p = Path(index_dir) / "passages.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"No index at {index_dir}. Run `build-index` first.")
    return [Passage.from_dict(d) for d in read_jsonl(p)]
