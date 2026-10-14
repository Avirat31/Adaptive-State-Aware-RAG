"""Deterministic, no-API, no-model generator: selects the best-matching evidence sentence(s) verbatim.

It performs NO reasoning/synthesis and cannot read yes/no from text (label stays None -> counted as abstention).
For multiple-choice it picks the option with the highest lexical overlap with the evidence (weak baseline).
"""
from __future__ import annotations

from ..schemas import RetrievalResult
from ..utils.text import split_sentences, tokenize
from .base import AnswerGenerator, GenerationOutput
from .citation_validation import validate_citations


class ExtractiveGenerator(AnswerGenerator):
    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: float = 0.3):
        self.max_sentences, self.min_overlap = max_sentences, min_overlap

    def generate(self, question, evidence, state, options=None) -> GenerationOutput:
        self.n_calls = 0
        targets = list(state.state.sub_questions) if state is not None and len(state.state.sub_questions) > 1 else [question]
        sents = [(s, r) for r in evidence for s in split_sentences(r.text)]
        stoks = [set(tokenize(s)) for s, _ in sents]
        chosen: list[int] = []
        for t in targets[: self.max_sentences]:
            tt = set(tokenize(t))
            if not tt:
                continue
            scored = sorted(((len(tt & st) / len(tt), -i, i) for i, st in enumerate(stoks) if i not in chosen), reverse=True)
            if scored and scored[0][0] >= self.min_overlap:
                chosen.append(scored[0][2])
        out = GenerationOutput(answer="", generator=self.name,
                               limitations=["Extractive baseline: copies evidence sentences; no synthesis or inference."])
        if not chosen:
            out.answer = "The retrieved evidence does not appear to contain sufficient information to answer this question."
            out.insufficient_evidence = True
        else:
            out.answer = " ".join(f"{sents[i][0]} [{sents[i][1].passage_id}]" for i in chosen)
            out.citations = [{"passage_id": sents[i][1].passage_id, "document_id": sents[i][1].document_id,
                              "claim_supported": sents[i][0]} for i in chosen]
        if options and evidence:
            ev = set(tokenize(" ".join(r.text for r in evidence)))
            sc = {k: (len(set(tokenize(v)) & ev) / max(1, len(set(tokenize(v))))) for k, v in options.items()}
            top = max(sc.values())
            winners = [k for k, v in sc.items() if v == top]
            if top > 0 and len(winners) == 1:
                out.label = winners[0]
            out.uncertainties.append("Option chosen by lexical overlap with retrieved text only.")
        if state is not None:
            if state.state.unresolved_sub_questions and len(state.state.sub_questions) > 0:
                out.uncertainties.append("Unresolved sub-questions: " + "; ".join(state.state.unresolved_sub_questions))
            for c in state.state.conflicting_claims[:3]:
                out.uncertainties.append(f"Possible (unverified) conflict between {c['passage_ids'][0]} and {c['passage_ids'][1]}.")
        return validate_citations(out, evidence)
