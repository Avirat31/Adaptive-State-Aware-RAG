from __future__ import annotations

from pathlib import Path

from ..schemas import Passage
from .bm25 import BM25Retriever
from .base import Retriever
from .dense import DenseRetriever, build_embedder
from .hybrid import HybridRetriever


def build_retriever(cfg: dict, passages: list[Passage], index_dir: Path | None, corpus_hash: str = "") -> Retriever:
    r = cfg["retrieval"]
    bm25 = BM25Retriever(passages) if r["method"] in ("bm25", "hybrid") else None
    dense = (DenseRetriever(passages, build_embedder(cfg), index_dir, corpus_hash)
             if r["method"] in ("dense", "hybrid") else None)
    if r["method"] == "bm25":
        return bm25
    if r["method"] == "dense":
        return dense
    return HybridRetriever(bm25, dense, r["rrf_k0"], r["top_k_sparse"], r["top_k_dense"])
