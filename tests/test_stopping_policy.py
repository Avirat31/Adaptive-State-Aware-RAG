from adaptive_biorag.memory.evidence_store import EvidenceStore
from adaptive_biorag.memory.stopping_policy import StoppingPolicy
from adaptive_biorag.schemas import RetrievalResult


def store(subs, texts):
    s = EvidenceStore("q", support_threshold=0.6)
    s.set_sub_questions(subs)
    for i, t in enumerate(texts):
        s.add_results([RetrievalResult("D", f"D::p{i}", t, "", "s", 1.0, "x", 1)], 0, 1, "q")
    return s


def test_stops_when_covered():
    d = StoppingPolicy().decide(store(["alpha beta?"], ["alpha beta"]), 1, 1, 1, 0)
    assert d.should_stop and d.reason == "sufficient_coverage" and d.evidence_coverage == 1.0


def test_stops_at_max_iterations_even_if_progressing():
    d = StoppingPolicy(max_iterations=2).decide(store(["alpha?", "gamma?"], ["alpha"]), 2, 5, 5, 0)
    assert d.should_stop and d.reason == "max_iterations_reached" and d.unresolved_items == ["gamma?"]


def test_stops_when_no_new_evidence_or_mostly_duplicates():
    p = StoppingPolicy(max_iterations=5)
    assert p.decide(store(["alpha?", "gamma?"], ["alpha"]), 1, 0, 3, 3).reason == "no_new_evidence"
    assert StoppingPolicy(5, minimum_new_evidence=1, max_duplicate_ratio=0.5).decide(store(["alpha?", "gamma?"], ["alpha"]), 1, 1, 4, 3).reason == "mostly_duplicates"


def test_continues_when_progress_and_unresolved():
    d = StoppingPolicy(max_iterations=5).decide(store(["alpha?", "gamma?"], ["alpha"]), 1, 2, 3, 1)
    assert not d.should_stop and d.reason == "continue_unresolved_subquestions"


def test_failure_stops():
    assert StoppingPolicy().decide(store(["a?"], []), 1, 0, 0, 0, failed=True).reason == "retrieval_failure"


def test_no_subquestions_case_requires_evidence():
    s = EvidenceStore("q")
    assert not StoppingPolicy(max_iterations=3).decide(s, 1, 1, 1, 0).should_stop   # m=0 and no evidence -> coverage 0
    assert s.coverage() == 0.0
    s.add_results([RetrievalResult("D", "D::p0", "t", "", "s", 1.0, "x", 1)], None, 1, "q")
    assert s.coverage() == 1.0


def test_loop_always_terminates_in_pipeline(cfg, passages):
    from adaptive_biorag.experiments.runner import build_pipeline
    from adaptive_biorag.schemas import Example
    cfg["routing"].update(enabled=False, forced_route="multi_step")
    for mem in (True, False):
        cfg["memory"]["enabled"] = mem
        p = build_pipeline(cfg, passages, None, "h").answer(Example("1", "Which drug inhibits kernase-7 encoded by KRN7?", "d", "s"))
        assert 1 <= p["n_iterations"] <= cfg["stopping"]["max_iterations"] and p["stop_reason"]
