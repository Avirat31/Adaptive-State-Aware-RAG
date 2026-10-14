"""AdaptiveBioRAG pipeline: route -> (single-step | iterative multi-step) -> assemble evidence -> generate -> validate."""
from __future__ import annotations

from ..evaluation.efficiency_metrics import Meter
from ..generation.base import AnswerGenerator
from ..logging_utils import get_logger
from ..memory.contradiction import ContradictionDetector, LexicalPolarityDetector
from ..memory.evidence_store import EvidenceStore
from ..memory.stopping_policy import StoppingPolicy
from ..retrieval.base import Retriever
from ..retrieval.reranker import Reranker
from ..routing.heuristic_router import Router
from ..schemas import Example, RetrievalResult
from .decomposer import QueryDecomposer

log = get_logger("pipeline")
MAX_QUESTION_CHARS = 2000


class Pipeline:
    def __init__(self, cfg: dict, retriever: Retriever, router: Router, decomposer: QueryDecomposer,
                 generator: AnswerGenerator, reranker: Reranker | None = None, llm_client=None,
                 detector: ContradictionDetector | None = None):
        self.cfg, self.retriever, self.router, self.decomposer = cfg, retriever, router, decomposer
        self.generator, self.reranker, self.llm = generator, reranker, llm_client
        r = cfg["retrieval"]
        self.top_k, self.final_k, self.rerank_n = r["top_k_fused"], r["final_evidence_k"], r["rerank_candidates"]
        self.use_memory = cfg["memory"]["enabled"]
        self.policy = StoppingPolicy.from_config(cfg)
        self.detector = detector if detector is not None else (LexicalPolarityDetector() if cfg["memory"]["detect_conflicts"] else None)

    # ------------------------------------------------------------------------------------------------------------------
    def _retrieve(self, query: str, meter: Meter) -> list[RetrievalResult]:
        with meter.time("retrieval"):
            depth = max(self.top_k, self.rerank_n) if self.reranker else self.top_k
            res = self.retriever.retrieve(query, depth)
        meter.count("retrieval_calls")
        meter.count("retrieved_passages", len(res))
        if self.reranker is not None:
            with meter.time("rerank"):
                res = self.reranker.rerank(query, res[: self.rerank_n], self.top_k)
            meter.count("rerank_calls")
        return res[: self.top_k]

    @staticmethod
    def _bridge_source(store: EvidenceStore, i: int, n_sub: int) -> list[tuple[str, str]]:
        """Evidence that already covers OTHER sub-questions; if none, the passages with highest support overall."""
        others = [x for x in store.items.values() if any(store.support(j, x) >= store.thr for j in range(n_sub) if j != i)]
        if not others:
            sc = sorted(((max(store.support(j, x) for j in range(n_sub)), x) for x in store.items.values()),
                        key=lambda a: -a[0])
            others = [x for s, x in sc if s > 0][:2]
        return [(x.result.title, x.result.text) for x in others[:3]]

    def _new_store(self, question: str) -> EvidenceStore:
        m, s = self.cfg["memory"], self.cfg["stopping"]
        return EvidenceStore(question, m["deduplicate"], m["subq_support_threshold"], s["min_evidence_passages"], self.detector)

    # ------------------------------------------------------------------------------------------------------------------
    def _single_step(self, question: str, meter: Meter, logs: list, errors: list):
        store = self._new_store(question)
        store.set_sub_questions([question])
        failed = False
        try:
            res = self._retrieve(question, meter)
        except Exception as e:                                   # noqa: BLE001 - recorded, run continues
            res, failed = [], True
            errors.append({"stage": "retrieval", "error": f"{type(e).__name__}: {e}"})
        before = store.coverage()
        st = store.add_results(res, 0, 1, question)
        store.state.iteration = 1
        cov = store.coverage()
        store.state.stop_reason = "single_step_complete" if not failed else "retrieval_failure"
        logs.append({"iteration": 1, "route": "single_step", "queries": [question], "retrieved_ids": [r.passage_id for r in res],
                     "new_evidence_ids": st["new_ids"], "duplicates": st["duplicates"], "coverage_before": before,
                     "coverage_after": cov, "stop_decision": {"should_stop": True, "reason": store.state.stop_reason,
                                                              "evidence_coverage": cov,
                                                              "unresolved_items": store.state.unresolved_sub_questions},
                     "acted_on_decision": True})
        return store, store.assemble(self.top_k)

    def _multi_step(self, question: str, meter: Meter, logs: list, errors: list):
        store = self._new_store(question)
        with meter.time("decomposition"):
            try:
                subs = self.decomposer.decompose(question, store)
            except Exception as e:                               # noqa: BLE001
                errors.append({"stage": "decomposition", "error": f"{type(e).__name__}: {e}"})
                subs = [question]
        subs = subs[: self.cfg["decomposition"]["max_sub_questions"]] or [question]
        store.set_sub_questions(subs)
        used_queries: set[str] = set()
        prev_round: list[RetrievalResult] = []
        for it in range(1, self.policy.max_iterations + 1):
            store.state.iteration = it
            targets = (list(range(len(subs))) if not self.use_memory else
                       [i for i, s in enumerate(subs) if s in store.state.unresolved_sub_questions]) or list(range(len(subs)))
            cov_before = store.coverage()
            queries, new_ids, dups, total, failed = [], [], 0, 0, False
            this_round: list[RetrievalResult] = []
            for i in targets:
                if it == 1:
                    q = subs[i]
                else:
                    if self.use_memory:    # evidence-aware: bridge terms from the evidence most relevant to the question
                        src = self._bridge_source(store, i, len(subs))
                    else:                  # memory-less: pseudo-relevance feedback from the previous round's top-2 results
                        src = [(r.title, r.text) for r in prev_round[:2]]
                    q = self.decomposer.follow_up_query(subs[i], question, src)
                if q in used_queries:
                    continue
                used_queries.add(q)
                queries.append(q)
                try:
                    res = self._retrieve(q, meter)
                except Exception as e:                           # noqa: BLE001
                    failed = True
                    errors.append({"stage": "retrieval", "query": q, "error": f"{type(e).__name__}: {e}"})
                    continue
                this_round += res
                st = store.add_results(res, i, it, q)
                new_ids += st["new_ids"]; dups += st["duplicates"]; total += st["total"]
            prev_round = this_round
            if not queries:                                      # every reformulation was already issued: nothing left to try
                cov = store.coverage()
                logs.append({"iteration": it, "route": "multi_step", "queries": [], "retrieved_ids": [], "new_evidence_ids": [],
                             "duplicates": 0, "coverage_before": cov_before, "coverage_after": cov,
                             "stop_decision": {"should_stop": True, "reason": "no_new_queries", "evidence_coverage": cov,
                                               "unresolved_items": list(store.state.unresolved_sub_questions)},
                             "acted_on_decision": True})
                store.state.stop_reason = "no_new_queries"
                break
            dec = self.policy.decide(store, it, len(new_ids), total, dups, failed)
            act = self.use_memory
            logs.append({"iteration": it, "route": "multi_step", "queries": queries, "retrieved_ids": [r.passage_id for r in this_round],
                         "new_evidence_ids": new_ids, "duplicates": dups, "coverage_before": cov_before,
                         "coverage_after": dec.evidence_coverage, "stop_decision": dec.to_dict(), "acted_on_decision": act,
                         "retrieval_config": {"method": self.cfg["retrieval"]["method"], "top_k": self.top_k,
                                              "reranker": self.reranker.name if self.reranker else None}})
            if act and dec.should_stop:
                store.state.stop_reason = dec.reason
                break
            if not act and (it == self.policy.max_iterations or failed):
                store.state.stop_reason = "fixed_iterations_complete" if not failed else "retrieval_failure"
        else:
            store.state.stop_reason = store.state.stop_reason or "max_iterations_reached"
        if store.state.stop_reason is None:
            store.state.stop_reason = "max_iterations_reached"
        return store, store.assemble(self.top_k)

    # ------------------------------------------------------------------------------------------------------------------
    def answer(self, example: Example, options: dict | None = None) -> dict:
        """Run one question. NOTE: only example.question (and options for MCQ) is used; gold fields are never read."""
        q = example.question
        if not isinstance(q, str) or not q.strip() or len(q) > MAX_QUESTION_CHARS:
            raise ValueError(f"Invalid question for id={example.question_id}: must be a non-empty string <= {MAX_QUESTION_CHARS} chars")
        meter, logs, errors = Meter(), [], []
        snap0 = self.llm.snapshot() if self.llm else None
        with meter.time("total"):
            route = self.router.route(q)
            if route.route == "multi_step":
                store, evidence = self._multi_step(q, meter, logs, errors)
            else:
                store, evidence = self._single_step(q, meter, logs, errors)
            ranked, evidence = evidence, evidence[: self.final_k]
            with meter.time("generation"):
                try:
                    out = self.generator.generate(q, evidence, store, options)
                except Exception as e:                           # noqa: BLE001
                    from ..generation.base import GenerationOutput
                    out = GenerationOutput("Answer generation failed; no answer produced.", insufficient_evidence=True,
                                           error=f"{type(e).__name__}: {e}", generator=self.generator.name)
                    errors.append({"stage": "generation", "error": out.error})
            meter.count("generation_calls", 1 if self.generator.name != "extractive" else 0)
        if self.llm:
            meter.add_llm_delta(snap0, self.llm.snapshot())
        eff = meter.summary()
        eff["unique_evidence_passages"] = len(store.items)
        eff["duplicate_retrievals"] = len(store.state.duplicate_passage_ids)
        cov = store.coverage()
        if cov < self.cfg["stopping"]["evidence_coverage_threshold"] and not out.insufficient_evidence and route.route == "multi_step":
            out.uncertainties.append(f"Evidence coverage of sub-questions was {cov:.2f}; the answer may be incomplete.")
        return {"question_id": example.question_id, "question": q, "dataset_name": example.dataset_name, "split": example.split,
                "question_type": example.question_type, "route": route.route, "routing": route.to_dict(),
                "sub_questions": store.state.sub_questions, "n_iterations": len(logs), "stop_reason": store.state.stop_reason,
                "evidence_coverage": cov, "unresolved_sub_questions": store.state.unresolved_sub_questions,
                "conflicting_claims": store.state.conflicting_claims,
                "retrieval_ranking": [{"rank": r.rank, "passage_id": r.passage_id, "document_id": r.document_id, "score": r.score,
                               "retriever": r.retriever, "alias_passages": [a["passage_id"] for a in r.metadata.get("aliases", [])],
                               "alias_documents": [a["document_id"] for a in r.metadata.get("aliases", [])]} for r in ranked],
                "final_evidence": [r.to_dict(include_text=True) for r in evidence],
                "answer": out.to_dict(), "iteration_logs": logs, "efficiency": eff, "errors": errors,
                "gold": {"gold_answer": example.gold_answer, "gold_document_ids": example.gold_document_ids,
                         "gold_passage_ids": example.gold_passage_ids, "options": options}}
