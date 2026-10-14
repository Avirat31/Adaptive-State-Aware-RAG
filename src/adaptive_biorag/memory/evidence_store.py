"""Evidence-state manager: structured memory of WHAT evidence was found and WHICH sub-questions remain unresolved."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..schemas import RetrievalResult
from ..utils.text import text_hash
from .contradiction import ContradictionDetector
from .coverage import compute_coverage, support_score, token_set


@dataclass
class EvidenceState:
    original_question: str
    sub_questions: list[str] = field(default_factory=list)
    covered_sub_questions: list[str] = field(default_factory=list)
    unresolved_sub_questions: list[str] = field(default_factory=list)
    retrieved_passages: list[str] = field(default_factory=list)      # unique passage IDs in first-seen order
    seen_passage_ids: list[str] = field(default_factory=list)
    duplicate_passage_ids: list[str] = field(default_factory=list)   # IDs seen again (or text-duplicates)
    conflicting_claims: list[dict] = field(default_factory=list)
    iteration: int = 0
    retrieval_history: list[dict] = field(default_factory=list)
    stop_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceItem:
    result: RetrievalResult
    first_iteration: int
    tokens: frozenset
    retrievers: set = field(default_factory=set)
    rank_by_subq: dict = field(default_factory=dict)       # sub-question index -> best rank it had for that query
    queries: list = field(default_factory=list)
    times_retrieved: int = 1                               # retrieval count; NOT independent corroboration


class EvidenceStore:
    def __init__(self, question: str, deduplicate: bool = True, support_threshold: float = 0.6, min_evidence: int = 1,
                 detector: ContradictionDetector | None = None):
        self.state = EvidenceState(original_question=question)
        self.items: dict[str, EvidenceItem] = {}
        self._hash_to_id: dict[str, str] = {}
        self.dedup_text, self.thr, self.min_evidence, self.detector = deduplicate, support_threshold, min_evidence, detector
        self._subq_tokens: list[frozenset] = []

    # -- sub-questions ----------------------------------------------------------------------------------------------
    def set_sub_questions(self, subs: list[str]) -> None:
        self.state.sub_questions = list(subs)
        self._subq_tokens = [token_set(s) for s in subs]
        self._refresh()

    def support(self, sq_idx: int, item: EvidenceItem) -> float:
        return support_score(self._subq_tokens[sq_idx], item.tokens)

    def _refresh(self) -> float:
        covered, unresolved = [], []
        for i, sq in enumerate(self.state.sub_questions):
            ok = any(self.support(i, it) >= self.thr for it in self.items.values())
            (covered if ok else unresolved).append(sq)
        self.state.covered_sub_questions, self.state.unresolved_sub_questions = covered, unresolved
        return compute_coverage(len(self.state.sub_questions), len(covered), len(self.items), self.min_evidence)

    def coverage(self) -> float:
        return self._refresh()

    # -- adding evidence ---------------------------------------------------------------------------------------------
    def add_results(self, results: list[RetrievalResult], sub_question_idx: int | None, iteration: int, query: str) -> dict:
        new_ids, dups, new_items = [], 0, []
        for r in results:
            pid, h = r.passage_id, text_hash(r.text)
            canon = pid if pid in self.items else (self._hash_to_id.get(h) if self.dedup_text else None)
            if canon is not None:                                  # duplicate: do NOT add a second evidence item
                it = self.items[canon]
                it.times_retrieved += 1
                it.retrievers.add(r.retriever)
                if sub_question_idx is not None:
                    it.rank_by_subq[sub_question_idx] = min(it.rank_by_subq.get(sub_question_idx, 10 ** 6), r.rank)
                dups += 1
                self.state.duplicate_passage_ids.append(pid)
                continue
            it = EvidenceItem(r, iteration, token_set(f"{r.title} {r.text}"), {r.retriever}, queries=[query])
            if sub_question_idx is not None:
                it.rank_by_subq[sub_question_idx] = r.rank
            self.items[pid] = it
            self._hash_to_id[h] = pid
            self.state.retrieved_passages.append(pid)
            self.state.seen_passage_ids.append(pid)
            new_ids.append(pid)
            new_items.append(it)
        if self.detector is not None:
            keys = list(self.items)
            for it in new_items:                                  # each pair is examined once (new item vs earlier items)
                earlier = [self.items[k] for k in keys[: keys.index(it.result.passage_id)]]
                self.state.conflicting_claims.extend(self.detector.detect(it, earlier))
        self.state.retrieval_history.append({"iteration": iteration, "query": query, "sub_question_idx": sub_question_idx,
                                             "retrieved_ids": [r.passage_id for r in results], "new_ids": new_ids,
                                             "duplicates": dups})
        return {"new_unique": len(new_ids), "duplicates": dups, "total": len(results), "new_ids": new_ids}

    # -- output ------------------------------------------------------------------------------------------------------
    def assemble(self, k: int) -> list[RetrievalResult]:
        """Final evidence: round-robin across sub-questions (supporting passages first), then fill by best overall rank."""
        chosen: list[str] = []
        n_sq = len(self.state.sub_questions)
        lists = []
        for i in range(n_sq):
            cands = [(0 if self.support(i, it) >= self.thr else 1, it.rank_by_subq.get(i, 10 ** 6), pid)
                     for pid, it in self.items.items() if self.support(i, it) >= self.thr or i in it.rank_by_subq]
            lists.append([pid for _, _, pid in sorted(cands)])
        pos = [0] * n_sq
        progressed = True
        while len(chosen) < k and progressed:
            progressed = False
            for i in range(n_sq):
                while pos[i] < len(lists[i]) and lists[i][pos[i]] in chosen:
                    pos[i] += 1
                if pos[i] < len(lists[i]) and len(chosen) < k:
                    chosen.append(lists[i][pos[i]])
                    pos[i] += 1
                    progressed = True
        if len(chosen) < k:
            rest = sorted((pid for pid in self.items if pid not in chosen),
                          key=lambda p: (min(self.items[p].rank_by_subq.values(), default=10 ** 6), p))
            chosen += rest[: k - len(chosen)]
        out = []
        for new_rank, pid in enumerate(chosen[:k], 1):
            r = self.items[pid].result
            out.append(RetrievalResult(r.document_id, r.passage_id, r.text, r.title, r.source, r.score, r.retriever, new_rank,
                                       {**r.metadata, "retrieved_times": self.items[pid].times_retrieved,
                                        "first_iteration": self.items[pid].first_iteration}))
        return out

    def summary(self) -> str:
        s = self.state
        lines = [f"Question: {s.original_question}",
                 f"Sub-questions covered: {len(s.covered_sub_questions)}/{len(s.sub_questions)}"]
        lines += [f"  UNRESOLVED: {u}" for u in s.unresolved_sub_questions]
        lines += [f"Unique evidence passages: {len(self.items)}; possible conflicts (unverified): {len(s.conflicting_claims)}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = self.state.to_dict()
        d["coverage"] = self.coverage()
        return d
