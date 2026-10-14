import pytest

from adaptive_biorag.evaluation.answer_metrics import evaluate_answers, macro_f1, normalize_answer, rouge_l_f, token_prf
from adaptive_biorag.evaluation.citation_metrics import evaluate_citations
from adaptive_biorag.evaluation.efficiency_metrics import Meter, aggregate_efficiency
from adaptive_biorag.evaluation.retrieval_metrics import evaluate_retrieval, mrr, ndcg_at_k, recall_at_k


def rk(*docs):
    return [{"rank": i, "passage_id": f"{d}::p0", "document_id": d, "alias_passages": [], "alias_documents": []} for i, d in enumerate(docs, 1)]


def pred(ranking, gold_docs, gold_ans=None, ans="", label=None, opts=None):
    return {"retrieval_ranking": ranking, "gold": {"gold_document_ids": gold_docs, "gold_passage_ids": [], "gold_answer": gold_ans, "options": opts},
            "answer": {"answer": ans, "label": label, "citations": [], "invalid_citations": []}, "final_evidence": [],
            "efficiency": {"latency_total_s": 0.1, "retrieval_calls": 1, "input_tokens": None, "output_tokens": None, "estimated_cost": None},
            "route": "single_step", "routing": {"complexity_score": 0.0}, "n_iterations": 1, "stop_reason": "x", "evidence_coverage": 1.0}


def test_recall_mrr_ndcg_values():
    sets = [{"a"}, {"x"}, {"b"}]
    gold = {"a", "b"}
    assert recall_at_k(sets, gold, 1) == 0.5 and recall_at_k(sets, gold, 3) == 1.0 and mrr(sets, gold) == 1.0
    assert mrr([{"x"}, {"a"}], gold) == 0.5
    assert ndcg_at_k(sets, gold, 3) == pytest.approx((1 + 0.5) / (1 + 1 / 1.5849625))  # DCG=1+1/log2(4); IDCG=1+1/log2(3)


def test_alias_documents_count_as_hits():
    r = [{"document_id": "D003", "passage_id": "D003::p0", "alias_documents": ["D012"], "alias_passages": ["D012::p0"]}]
    res = evaluate_retrieval([pred(r, ["D012"])], [1])
    assert res["doc_metrics"]["recall@1"] == 1.0


def test_missing_gold_excluded_not_failure():
    res = evaluate_retrieval([pred(rk("a"), ["a"]), pred(rk("z"), [])], [1, 5])
    assert res["n_with_gold_doc"] == 1 and res["n_missing_gold_doc"] == 1 and res["doc_metrics"]["recall@1"] == 1.0
    assert res["passage_metrics"] is None                       # no passage-level gold -> not reported
    assert evaluate_retrieval([pred(rk("z"), [])], [1])["doc_metrics"] is None


def test_answer_metrics_text_and_classification():
    assert normalize_answer("The Kernase-7!") == "kernase 7"
    assert token_prf("valtrexin drug", "valtrexin")[2] == pytest.approx(2 / 3) and rouge_l_f("x y z", "x z") == pytest.approx(0.8)
    r = evaluate_answers([pred([], [], [["valtrexin", "val"]], "Valtrexin [D1::p0]"), pred([], [], "yes", "Yes, it is.", None),
                          pred([], [], "no", "unclear", None), pred([], [], None, "x"),
                          pred([], [], "B", "x", "B", {"A": "p", "B": "q"})])
    assert r["n_no_gold"] == 1 and r["text_em"] == 1.0 and r["n_classification"] == 3
    assert r["cls_accuracy"] == pytest.approx(2 / 3) and r["cls_abstained"] == 1
    assert macro_f1(["yes", "no"], ["yes", None]) == pytest.approx(0.5)


def test_citation_metrics_handle_missing_and_insufficient():
    p = pred([], [], None, "Claim one [D1::p0]. Claim two.")
    p["answer"]["citations"] = [{"passage_id": "D1::p0", "valid": True}, {"passage_id": "X", "valid": False}]
    p["final_evidence"] = [{"passage_id": "D1::p0", "text": "claim one supported text"}]
    q = pred([], [], None, "no info"); q["answer"]["insufficient_evidence"] = True
    c = evaluate_citations([p, q])
    assert c["citation_validity"] == 0.5 and c["citation_coverage"] == 0.5 and c["n_insufficient_evidence_answers"] == 1
    assert evaluate_citations([])["citation_validity"] is None


def test_efficiency_never_fabricates_tokens_or_cost():
    m = Meter()
    with m.time("total"):
        m.count("retrieval_calls", 2)
    s = m.summary()
    assert s["retrieval_calls"] == 2 and s["estimated_cost"] is None and s["llm_calls"] == 0
    agg = aggregate_efficiency([{**s, "input_tokens": None}])
    assert agg["input_tokens_sum"] is None and agg["estimated_cost_sum"] is None


def test_claim_segmentation_attaches_trailing_citation():
    from adaptive_biorag.evaluation.citation_metrics import split_claims
    assert split_claims("First claim. [A::p0] Second claim. [B::p1]") == ["First claim. [A::p0]", "Second claim. [B::p1]"]
