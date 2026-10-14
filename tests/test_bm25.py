from adaptive_biorag.retrieval.bm25 import BM25Retriever


def test_ranking_and_fields(passages):
    r = BM25Retriever(passages)
    res = r.retrieve("Which drug inhibits kernase-7?", 3)
    assert res[0].passage_id == "B::p0" and [x.rank for x in res] == list(range(1, len(res) + 1))
    assert res[0].retriever == "bm25" and res[0].document_id == "B" and res[0].score > res[-1].score


def test_empty_and_unknown_queries(passages):
    r = BM25Retriever(passages)
    assert r.retrieve("", 5) == [] and r.retrieve("zzzzqqqq", 5) == []
    assert BM25Retriever([]).retrieve("x", 3) == []


def test_top_k_and_determinism(passages):
    r = BM25Retriever(passages)
    assert len(r.retrieve("kernase-7 gene", 1)) == 1
    assert [x.passage_id for x in r.retrieve("kernase-7", 4)] == [x.passage_id for x in r.retrieve("kernase-7", 4)]
