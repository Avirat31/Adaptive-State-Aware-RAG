import pytest

from adaptive_biorag.retrieval.hybrid import HybridRetriever
from adaptive_biorag.retrieval.reranker import LexicalReranker
from adaptive_biorag.retrieval.rrf import rrf_fuse
from adaptive_biorag.schemas import RetrievalResult


def R(pid, score=1.0, name="x", rank=0):
    return RetrievalResult(pid.split("::")[0], pid, f"text {pid}", "t", "s", score, name, rank)


def test_rrf_formula_and_order():
    out = rrf_fuse({"a": [R("1::p"), R("2::p"), R("3::p")], "b": [R("2::p"), R("1::p")]}, k0=60)
    s = {r.passage_id: r.score for r in out}
    assert s["1::p"] == pytest.approx(1 / 61 + 1 / 62) and s["2::p"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["3::p"] == pytest.approx(1 / 63) and out[-1].passage_id == "3::p"
    assert [r.rank for r in out] == [1, 2, 3]


def test_rrf_dedup_provenance_and_missing_retriever():
    out = rrf_fuse({"bm25": [R("1::p", 9.0), R("2::p", 5.0)], "dense": []}, 60)
    assert len(out) == 2 and out[0].metadata["ranks"] == {"bm25": 1} and out[0].metadata["scores"]["bm25"] == 9.0
    assert rrf_fuse({"a": [], "b": []}) == []
    both = rrf_fuse({"a": [R("1::p", 2.0)], "b": [R("1::p", 0.3)]}, 60)
    assert len(both) == 1 and both[0].metadata["ranks"] == {"a": 1, "b": 1}


def test_rrf_ignores_score_scale_and_validates():
    big = rrf_fuse({"a": [R("1::p", 1e6), R("2::p", 1e5)], "b": [R("2::p", 0.9), R("1::p", 0.1)]}, 60)
    assert big[0].passage_id == "1::p"                          # tie broken deterministically by id, not by raw scores
    with pytest.raises(ValueError):
        rrf_fuse({"a": []}, k0=0)
    assert len(rrf_fuse({"a": [R(f"{i}::p") for i in range(10)]}, 60, top_k=3)) == 3


def test_hybrid_with_one_empty_retriever(passages):
    class Empty:
        def retrieve(self, q, k): return []
    from adaptive_biorag.retrieval.bm25 import BM25Retriever
    h = HybridRetriever(BM25Retriever(passages), Empty(), 60)
    res = h.retrieve("valtrexin kernase-7", 3)
    assert res and res[0].passage_id == "B::p0" and res[0].retriever == "hybrid_rrf"


def test_reranker_preserves_original_rank_and_score():
    cands = [R("1::p", 3.0, "hybrid_rrf", 1), R("2::p", 2.0, "hybrid_rrf", 2)]
    cands[1].text = "kernase-7 inhibitor valtrexin"
    out = LexicalReranker().rerank("valtrexin kernase-7 inhibitor", cands)
    assert out[0].passage_id == "2::p" and out[0].metadata["pre_rerank_rank"] == 2 and out[0].metadata["pre_rerank_score"] == 2.0
    assert LexicalReranker().rerank("q", []) == []
