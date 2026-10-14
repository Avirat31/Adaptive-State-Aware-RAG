from __future__ import annotations

from ..schemas import RetrievalResult
from .base import Retriever
from .rrf import rrf_fuse


class HybridRetriever(Retriever):
    name = "hybrid_rrf"

    def __init__(self, sparse: Retriever, dense: Retriever, k0: int = 60, top_k_sparse: int = 50, top_k_dense: int = 50):
        self.sparse, self.dense, self.k0, self.ks, self.kd = sparse, dense, k0, top_k_sparse, top_k_dense

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return rrf_fuse({"bm25": self.sparse.retrieve(query, self.ks), "dense": self.dense.retrieve(query, self.kd)},
                        self.k0, top_k)
