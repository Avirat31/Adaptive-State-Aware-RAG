import json

import pytest

from adaptive_biorag.datasets import DatasetError, get_adapter, validate_examples
from adaptive_biorag.schemas import Example


def test_synthetic_loads_and_validates(cfg):
    ex = get_adapter(cfg).load()
    rep = validate_examples(ex)
    assert rep["n_examples"] == 9 and rep["missing"]["gold_document_ids"] == 1 and rep["evidence_retrieval_eval_available"]


def test_duplicate_ids_and_empty_questions_rejected():
    e = Example("1", "q?", "d", "s")
    with pytest.raises(DatasetError, match="duplicate"):
        validate_examples([e, Example("1", "other?", "d", "s")])
    with pytest.raises(DatasetError, match="empty"):
        validate_examples([Example("2", "  ", "d", "s")])


def test_duplicate_question_text_reported_not_fatal():
    rep = validate_examples([Example("1", "Same?", "d", "s"), Example("2", "same?", "d", "s")])
    assert rep["duplicate_question_texts"] == 1


def test_missing_file_gives_helpful_error(cfg, tmp_path):
    cfg["dataset"].update(name="pubmedqa", path=str(tmp_path / "x.json"))
    with pytest.raises(DatasetError, match="not found"):
        get_adapter(cfg).load()


def test_pubmedqa_adapter_no_context_leak(cfg, tmp_path):
    p = tmp_path / "pq.json"
    p.write_text(json.dumps({"111": {"QUESTION": "Does X help?", "CONTEXTS": ["Abstract text."], "final_decision": "Yes", "LONG_ANSWER": "L"}}))
    cfg["dataset"].update(name="pubmedqa", path=str(p), split="test")
    a = get_adapter(cfg)
    ex = a.load()[0]
    assert ex.gold_answer == "yes" and ex.gold_document_ids == ["111"] and "contexts" not in ex.metadata
    assert a.corpus_documents()[0].text == "Abstract text."
    cfg["dataset"]["provide_context"] = True
    assert "contexts" in get_adapter(cfg).load()[0].metadata


def test_pubmedqa_bad_schema(cfg, tmp_path):
    p = tmp_path / "pq.json"
    p.write_text(json.dumps({"1": {"QUESTION": "q"}}))
    cfg["dataset"].update(name="pubmedqa", path=str(p))
    with pytest.raises(DatasetError, match="final_decision"):
        get_adapter(cfg).load()


def test_bioasq_adapter_missing_annotations_not_fabricated(cfg, tmp_path):
    p = tmp_path / "bq.json"
    p.write_text(json.dumps({"questions": [
        {"id": "q1", "body": "Is X?", "type": "yesno", "exact_answer": "yes", "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/123"],
         "snippets": [{"document": "http://www.ncbi.nlm.nih.gov/pubmed/123", "text": "snip"}]},
        {"id": "q2", "body": "Test question?", "type": "factoid"}]}))
    cfg["dataset"].update(name="bioasq", path=str(p))
    a = get_adapter(cfg)
    ex = a.load()
    assert ex[0].gold_document_ids == ["123"] and ex[0].gold_passage_ids == []
    assert ex[1].gold_answer is None and ex[1].gold_document_ids == [] and ex[1].metadata["no_gold_annotations"]
    assert a.corpus_documents()[0].document_id == "123"


def test_mirage_requires_valid_subset(cfg, tmp_path):
    p = tmp_path / "benchmark.json"
    p.write_text(json.dumps({"mmlu": {"mmlu_0": {"question": "Q?", "options": {"A": "x", "B": "y"}, "answer": "B"}}}))
    cfg["dataset"].update(name="mirage", path=str(p), subset="nope")
    with pytest.raises(DatasetError, match="available subsets"):
        get_adapter(cfg).load()
    cfg["dataset"]["subset"] = "mmlu"
    ex = get_adapter(cfg).load()[0]
    assert ex.gold_answer == "B" and ex.metadata["options"]["A"] == "x" and not ex.has_gold_evidence


def test_medhopqa_field_map(cfg, tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"qid": "1", "q_text": "Hop?", "ans": "z"}) + "\n")
    cfg["dataset"].update(name="medhopqa", path=str(p))
    with pytest.raises(DatasetError, match="field_map"):
        get_adapter(cfg).load()
    cfg["dataset"]["field_map"] = {"question": "q_text", "answer": "ans", "id": "qid"}
    ex = get_adapter(cfg).load()[0]
    assert ex.question == "Hop?" and ex.gold_answer == "z" and ex.question_id == "1"


def test_malformed_jsonl_reports_line(cfg, tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"question_id": "1", "question": "ok?"}\n{bad json\n')
    cfg["dataset"].update(name="jsonl", path=str(p))
    with pytest.raises(ValueError, match=":2:"):
        get_adapter(cfg).load()
