from adaptive_biorag.memory.contradiction import LexicalPolarityDetector, polarity
from adaptive_biorag.memory.coverage import compute_coverage
from adaptive_biorag.memory.evidence_store import EvidenceStore
from adaptive_biorag.schemas import RetrievalResult


def R(pid, text, doc=None, rank=1, title="t"):
    return RetrievalResult(doc or pid.split("::")[0], pid, text, title, "s", 1.0, "hybrid_rrf", rank)


def test_add_dedup_by_id_and_text_and_no_inflation():
    s = EvidenceStore("q", deduplicate=True, support_threshold=0.5)
    s.set_sub_questions(["alpha beta gene?"])
    a = s.add_results([R("A::p0", "alpha beta gene text"), R("B::p0", "other words")], 0, 1, "q1")
    assert a == {"new_unique": 2, "duplicates": 0, "total": 2, "new_ids": ["A::p0", "B::p0"]}
    b = s.add_results([R("A::p0", "alpha beta gene text"), R("C::p0", "Alpha  beta GENE text")], 0, 2, "q2")
    assert b["new_unique"] == 0 and b["duplicates"] == 2 and len(s.items) == 2          # id dup + text dup
    assert s.items["A::p0"].times_retrieved == 3 and s.state.duplicate_passage_ids == ["A::p0", "C::p0"]
    assert s.coverage() == 1.0


def test_text_dedup_can_be_disabled_but_id_dedup_stays():
    s = EvidenceStore("q", deduplicate=False)
    s.add_results([R("A::p0", "same text"), R("B::p0", "same text"), R("A::p0", "same text")], None, 1, "q")
    assert set(s.items) == {"A::p0", "B::p0"}


def test_coverage_and_unresolved_tracking():
    s = EvidenceStore("q", support_threshold=0.6)
    s.set_sub_questions(["kinase gene mutation?", "drug inhibitor assay?"])
    assert s.coverage() == 0.0 and len(s.state.unresolved_sub_questions) == 2
    s.add_results([R("A::p0", "the kinase gene mutation was found")], 0, 1, "q")
    assert s.coverage() == 0.5 and s.state.unresolved_sub_questions == ["drug inhibitor assay?"]
    s.add_results([R("B::p0", "a drug inhibitor in assay")], 1, 2, "q")
    assert s.coverage() == 1.0 and s.state.unresolved_sub_questions == []


def test_coverage_zero_subquestion_definition():
    assert compute_coverage(0, 0, 0) == 0.0 and compute_coverage(0, 0, 2, 1) == 1.0 and compute_coverage(4, 3, 9) == 0.75


def test_assemble_round_robin_and_serialisation():
    s = EvidenceStore("q", support_threshold=0.6)
    s.set_sub_questions(["alpha one?", "beta two?"])
    s.add_results([R("A1::p", "alpha one", rank=1), R("A2::p", "alpha one again", rank=2)], 0, 1, "q")
    s.add_results([R("B1::p", "beta two", rank=1)], 1, 1, "q")
    ids = [r.passage_id for r in s.assemble(2)]
    assert "B1::p" in ids and "A1::p" in ids
    d = s.to_dict()
    assert d["coverage"] == 1.0 and "retrieval_history" in d and s.summary().startswith("Question")


def test_conflict_detection_is_conservative_and_optional():
    a = R("D8::p0", "Beta-minerol did not significantly reduce fracture risk in a randomized trial.")
    b = R("D9::p0", "Beta-minerol supplementation significantly reduced fracture risk in a cohort study.")
    assert polarity(a.text) == "negative" and polarity(b.text) == "positive"
    s = EvidenceStore("q", detector=LexicalPolarityDetector())
    s.add_results([a, b], None, 1, "q")
    c = s.state.conflicting_claims
    assert len(c) == 1 and c[0]["status"] == "possible_conflict" and set(c[0]["document_ids"]) == {"D8", "D9"}
    s2 = EvidenceStore("q")                                   # detector disabled -> pipeline still works
    s2.add_results([a, b], None, 1, "q")
    assert s2.state.conflicting_claims == []
    s3 = EvidenceStore("q", detector=LexicalPolarityDetector())
    s3.add_results([a, R("D8::p1", "Beta-minerol did not reduce fracture risk.", doc="D8")], None, 1, "q")
    assert s3.state.conflicting_claims == []                  # same document -> never flagged
