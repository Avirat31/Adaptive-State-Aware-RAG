"""Optional rerankers. Original retrieval rank/score are preserved in result.metadata."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import RetrievalResult
from ..utils.text import tokenize


class Reranker(ABC):
    name = "reranker"

    @abstractmethod
    def score(self, query: str, results: list[RetrievalResult]) -> list[float]: ...

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        if not results:
            return []
        scores = self.score(query, results)
        if len(scores) != len(results):
            raise RuntimeError(f"{self.name} returned {len(scores)} scores for {len(results)} candidates")
        order = sorted(range(len(results)), key=lambda i: (-scores[i], results[i].rank, results[i].passage_id))
        out = []
        for new_rank, i in enumerate(order[: top_k or len(results)], 1):
            r = results[i]
            md = dict(r.metadata)
            md.update({"pre_rerank_rank": r.rank, "pre_rerank_score": r.score, "pre_rerank_retriever": r.retriever,
                       "rerank_score": float(scores[i]), "reranker": self.name})
            out.append(RetrievalResult(r.document_id, r.passage_id, r.text, r.title, r.source, float(scores[i]),
                                       f"{r.retriever}+{self.name}", new_rank, md))
        return out


class LexicalReranker(Reranker):
    """Offline, deterministic: fraction of query terms covered (title weighted x2) + small bigram bonus.
    A cheap stand-in for a cross-encoder so the reranking code path runs without model weights."""
    name = "lexical_rerank"

    def score(self, query, results):
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(results)
        qs = tokenize(query)
        qbi = set(zip(qs, qs[1:]))
        out = []
        for r in results:
            body, title = tokenize(r.text), set(tokenize(r.title))
            bs = set(body)
            cover = len(q & bs) / len(q) + 0.5 * len(q & title) / len(q)
            bi = len(qbi & set(zip(body, body[1:]))) / max(1, len(qbi))
            out.append(cover + 0.25 * bi)
        return out


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, batch_size: int = 16):
        self.name = f"ce:{model_name.split('/')[-1]}"
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError("reranker_method=cross_encoder needs `pip install sentence-transformers`.") from e
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(f"Could not load cross-encoder '{model_name}': {e}") from e
        self.bs = batch_size

    def score(self, query, results):
        try:
            return [float(s) for s in self.model.predict([(query, f"{r.title}. {r.text}") for r in results],
                                                         batch_size=self.bs, show_progress_bar=False)]
        except Exception as e:
            raise RuntimeError(f"Cross-encoder inference failed: {e}") from e


def build_reranker(cfg: dict) -> Reranker | None:
    r = cfg["retrieval"]
    if not r["use_reranker"]:
        return None
    return LexicalReranker() if r["reranker_method"] == "lexical" else CrossEncoderReranker(r["reranker_model"])
