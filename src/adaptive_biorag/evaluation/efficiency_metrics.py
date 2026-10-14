from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class Meter:
    """Per-question timers/counters. Unmeasurable quantities stay None (never fabricated)."""
    def __init__(self):
        self.t: dict[str, float] = defaultdict(float)
        self.c: dict[str, int] = defaultdict(int)
        self.llm = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}

    @contextmanager
    def time(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] += time.perf_counter() - t0

    def count(self, name: str, n: int = 1):
        self.c[name] += n

    def add_llm_delta(self, before: dict, after: dict):
        self.llm["calls"] += after["calls"] - before["calls"]
        for k in ("in", "out", "cost"):
            self.llm[k] = None if (self.llm[k] is None or after[k] is None or before[k] is None) else self.llm[k] + after[k] - before[k]

    def summary(self) -> dict:
        return {"latency_total_s": round(self.t["total"], 6), "latency_retrieval_s": round(self.t["retrieval"], 6),
                "latency_rerank_s": round(self.t["rerank"], 6), "latency_generation_s": round(self.t["generation"], 6),
                "latency_decomposition_s": round(self.t["decomposition"], 6),
                "retrieval_calls": self.c["retrieval_calls"], "rerank_calls": self.c["rerank_calls"],
                "retrieved_passages": self.c["retrieved_passages"], "generation_calls": self.c["generation_calls"],
                "llm_calls": self.llm["calls"], "input_tokens": self.llm["in"], "output_tokens": self.llm["out"],
                "estimated_cost": self.llm["cost"] if self.llm["calls"] else None}


def aggregate_efficiency(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    for k in ("latency_total_s", "latency_retrieval_s", "latency_rerank_s", "latency_generation_s", "retrieval_calls",
              "rerank_calls", "retrieved_passages", "unique_evidence_passages", "generation_calls", "llm_calls"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_median"] = float(np.median(vals))
            out[f"{k}_p95"] = float(np.percentile(vals, 95))
            out[f"{k}_sum"] = float(np.sum(vals))
    for k in ("input_tokens", "output_tokens", "estimated_cost"):
        vals = [r.get(k) for r in rows]
        out[f"{k}_sum"] = None if (not vals or any(v is None for v in vals)) else float(sum(vals))
    return out
