import pytest

from adaptive_biorag.corpus.chunker import chunk_document
from adaptive_biorag.corpus.deduplication import deduplicate_passages
from adaptive_biorag.corpus.index_manifest import build_corpus_index, build_passages
from adaptive_biorag.corpus.loader import load_documents_jsonl, normalize_document
from adaptive_biorag.schemas import Document


def doc(i, n):
    return Document(i, "T", " ".join(f"w{k}" for k in range(n)))


def test_stable_ids_and_overlap_and_offsets():
    d = doc("D1", 30)
    a, b = chunk_document(d, 10, 4), chunk_document(d, 10, 4)
    assert [p.passage_id for p in a] == [p.passage_id for p in b] == [f"D1::p{i}" for i in range(len(a))]
    assert a[0].text.split()[-4:] == a[1].text.split()[:4]
    for p in a:
        assert d.text[p.start_offset:p.end_offset] == p.text
    assert a[-1].text.endswith("w29")


def test_short_empty_and_bad_params():
    assert chunk_document(Document("E", "", ""), 10, 2) == []
    assert chunk_document(Document("S", "", "a b"), 10, 2, min_words=3) == []
    with pytest.raises(ValueError):
        chunk_document(doc("X", 10), 5, 5)


def test_normalize_document_failures():
    assert normalize_document({"document_id": "1", "text": "  "})[1] == "empty or missing text"
    assert normalize_document({"text": "x"})[1] == "missing document_id"
    assert normalize_document("str")[1] == "record is not an object"
    d, _ = normalize_document({"id": 5, "abstract": "Some  text\u00a0here"})
    assert d.document_id == "5" and d.text == "Some text here"


def test_dedup_keeps_provenance():
    ps, _ = build_passages([Document("A", "", "same text here ok"), Document("B", "", "Same  text here OK")], 50, 0, 3)
    assert len(ps) == 1 and ps[0].metadata["aliases"] == [{"passage_id": "B::p0", "document_id": "B"}]
    kept, rep = deduplicate_passages([])
    assert kept == [] and rep["n_duplicates_removed"] == 0


def test_loader_records_failures(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"document_id":"1","text":"ok text"}\n{"document_id":"1","text":"dup"}\n{"document_id":"2","text":""}\n')
    docs, fails = load_documents_jsonl(p)
    assert len(docs) == 1 and len(fails) == 2


def test_index_build_and_reuse(cfg):
    p1, m1, idx = build_corpus_index(cfg)
    p2, m2, _ = build_corpus_index(cfg)
    assert m1["corpus_hash"] == m2["corpus_hash"] and (idx / "passages.jsonl").exists()
    assert m1["n_documents"] == 12 and len(m1["load_failures"]) == 1 and m1["dedup"]["n_duplicates_removed"] == 1
    assert any(a["document_id"] == "D012" for p in p1 for a in p.metadata.get("aliases", []))
