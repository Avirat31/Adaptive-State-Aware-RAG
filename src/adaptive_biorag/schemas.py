"""Core shared data structures. Every module exchanges these types (no ad-hoc dicts across module boundaries)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Example:
    question_id: str
    question: str
    dataset_name: str
    split: str
    question_type: str | None = None
    gold_answer: Any = None
    gold_document_ids: list[str] = field(default_factory=list)
    gold_passage_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(question_id=str(d["question_id"]), question=d["question"], dataset_name=d.get("dataset_name", "unknown"),
                   split=d.get("split", "unknown"), question_type=d.get("question_type"), gold_answer=d.get("gold_answer"),
                   gold_document_ids=[str(x) for x in d.get("gold_document_ids") or []],
                   gold_passage_ids=[str(x) for x in d.get("gold_passage_ids") or []], metadata=dict(d.get("metadata") or {}))

    @property
    def has_gold_evidence(self) -> bool:
        return bool(self.gold_document_ids or self.gold_passage_ids)


@dataclass
class Document:
    document_id: str
    title: str
    text: str
    source: str = "unknown"
    url: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Passage:
    passage_id: str
    document_id: str
    text: str
    title: str = ""
    source: str = "unknown"
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Passage":
        return cls(**{k: d.get(k) for k in ("passage_id", "document_id", "text")}, title=d.get("title", ""),
                   source=d.get("source", "unknown"), start_offset=d.get("start_offset"), end_offset=d.get("end_offset"),
                   metadata=dict(d.get("metadata") or {}))

    @property
    def index_text(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class RetrievalResult:
    """Uniform result for ALL retrievers. `score` is retriever-specific and NOT comparable across retrievers."""
    document_id: str
    passage_id: str
    text: str
    title: str
    source: str
    score: float
    retriever: str
    rank: int = 0                       # 1-based rank within the producing retriever's list
    metadata: dict = field(default_factory=dict)

    def to_dict(self, include_text: bool = True) -> dict:
        d = asdict(self)
        if not include_text:
            d.pop("text")
        return d

    @classmethod
    def from_passage(cls, p: Passage, score: float, retriever: str, rank: int) -> "RetrievalResult":
        return cls(p.document_id, p.passage_id, p.text, p.title, p.source, float(score), retriever, rank,
                   {"aliases": p.metadata.get("aliases", [])})


@dataclass
class RouteDecision:
    route: str                       # "single_step" | "multi_step"
    complexity_score: float          # heuristic score in [0,1]; NOT a calibrated probability
    reason_codes: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    method: str = "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StopDecision:
    should_stop: bool
    reason: str
    evidence_coverage: float
    unresolved_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
