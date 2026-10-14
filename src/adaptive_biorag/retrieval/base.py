from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import RetrievalResult


class Retriever(ABC):
    """All retrievers return list[RetrievalResult] sorted best-first with 1-based `rank`."""
    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        ...
