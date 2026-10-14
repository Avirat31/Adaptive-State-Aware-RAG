from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from ..schemas import RetrievalResult


@dataclass
class GenerationOutput:
    answer: str
    citations: list[dict] = field(default_factory=list)       # {passage_id, document_id, claim_supported, valid}
    limitations: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False
    label: str | None = None                                   # option letter / yes|no|maybe when the task needs one
    invalid_citations: list[str] = field(default_factory=list)
    generator: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class AnswerGenerator(ABC):
    name = "generator"
    n_calls = 0

    @abstractmethod
    def generate(self, question: str, evidence: list[RetrievalResult], state, options: dict | None = None) -> GenerationOutput:
        ...
