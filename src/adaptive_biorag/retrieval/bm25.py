"""Self-contained Okapi BM25 (inverted index, deterministic tie-breaking). Rebuilt in-memory from passages.jsonl."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from ..schemas import Passage, RetrievalResult
from ..utils.text import tokenize
from .base import Retriever


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages, self.k1, self.b = list(passages), k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lens = np.zeros(len(self.passages))
        for i, p in enumerate(self.passages):
            tf = Counter(tokenize(p.index_text))
            lens[i] = sum(tf.values())
            for t, c in tf.items():
                self.postings[t].append((i, c))
        self.doc_len = lens
        self.avgdl = float(lens.mean()) if len(lens) and lens.mean() > 0 else 1.0
        n = len(self.passages)
        self.idf = {t: math.log(1.0 + (n - len(pl) + 0.5) / (len(pl) + 0.5)) for t, pl in self.postings.items()}

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.passages[kv[0]].passage_id))[:top_k]
        return [RetrievalResult.from_passage(self.passages[i], s, self.name, r) for r, (i, s) in enumerate(ranked, 1)]
