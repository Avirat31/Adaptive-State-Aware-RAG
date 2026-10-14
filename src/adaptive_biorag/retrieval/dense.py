"""Dense retrieval: exact cosine search over L2-normalised embeddings held in a NumPy matrix.

Memory: N_passages x dim x 4 bytes (float32). E.g. 1M passages x 1024 dims ~= 4 GB -> move to FAISS/HNSW at that scale.

Backends
- hashing: deterministic signed feature-hashing of word uni/bi-grams + char tri-grams. NO model download, CPU only.
  It is a *lexical-overlap-with-fuzzy-matching* stand-in that exercises the dense code path offline. It is NOT a
  semantic embedding model and results from it must not be reported as dense-retrieval results.
- sentence_transformers: any pretrained SentenceTransformer (e.g. BAAI/bge-m3). Downloads weights on first use
  (bge-m3 is ~2.2 GB on disk) - see README.
"""
from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger
from ..schemas import Passage, RetrievalResult
from ..utils.text import tokenize
from .base import Retriever

log = get_logger("dense")


class Embedder(ABC):
    backend: str = "embedder"
    model_name: str = ""

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    @property
    def signature(self) -> str:
        return f"{self.backend}:{self.model_name}"


class HashingEmbedder(Embedder):
    backend = "hashing"

    def __init__(self, dim: int = 512):
        self.dim, self.model_name = dim, f"dim{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = list(toks) + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            if len(t) >= 4:
                feats += [f"#{t[i:i + 3]}" for i in range(len(t) - 2)]
        for f in feats:
            h = zlib.crc32(f.encode("utf-8"))
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _enc(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)

    encode_queries = _enc
    encode_passages = _enc


class SentenceTransformerEmbedder(Embedder):
    backend = "sentence_transformers"

    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = "", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("retrieval.dense_backend=sentence_transformers needs `pip install sentence-transformers` "
                               "(install PyTorch first; see README).") from e
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Could not load embedding model '{model_name}': {e}. Check the name, network access "
                               f"and free disk space (weights are downloaded and cached on first use).") from e
        self.model_name, self.qp, self.pp, self.bs = model_name, query_prefix, passage_prefix, batch_size

    def _enc(self, texts, prefix):
        return np.asarray(self.model.encode([prefix + t for t in texts], batch_size=self.bs, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)

    def encode_queries(self, texts):
        return self._enc(texts, self.qp)

    def encode_passages(self, texts):
        return self._enc(texts, self.pp)


def build_embedder(cfg: dict) -> Embedder:
    r = cfg["retrieval"]
    if r["dense_backend"] == "hashing":
        return HashingEmbedder(r["dense_dim"])
    return SentenceTransformerEmbedder(r["dense_model"], r["query_prefix"], r["passage_prefix"], r["dense_batch_size"])


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, passages: list[Passage], embedder: Embedder, cache_dir: Path | None = None, corpus_hash: str = "",
                 batch_size: int = 64):
        self.passages, self.embedder = list(passages), embedder
        self.matrix = self._load_or_embed(cache_dir, corpus_hash, batch_size)

    def _cache_path(self, cache_dir, corpus_hash):
        if cache_dir is None:
            return None
        sig = zlib.crc32(f"{self.embedder.signature}|{corpus_hash}".encode())
        return Path(cache_dir) / f"dense_{self.embedder.backend}_{sig:08x}.npy"

    def _load_or_embed(self, cache_dir, corpus_hash, batch_size) -> np.ndarray:
        path = self._cache_path(cache_dir, corpus_hash)
        if path is not None and path.exists():
            m = np.load(path)
            if m.shape[0] == len(self.passages):
                return m
            log.warning("Stale embedding cache %s ignored", path)
        texts = [p.index_text for p in self.passages]
        parts = [self.embedder.encode_passages(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        m = np.concatenate(parts) if parts else np.zeros((0, 1), np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, m)
        return m

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not query.strip() or len(self.passages) == 0:
            return []
        q = self.embedder.encode_queries([query])[0]
        sims = self.matrix @ q
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), self.passages[i].passage_id))[:top_k]
        return [RetrievalResult.from_passage(self.passages[i], float(sims[i]), self.name, r)
                for r, i in enumerate(order, 1) if sims[i] > 0]
