import json

from adaptive_biorag.generation.base import GenerationOutput
from adaptive_biorag.generation.citation_validation import extract_inline_citations, validate_citations
from adaptive_biorag.generation.extractive import ExtractiveGenerator
from adaptive_biorag.generation.llm_provider import LLMError, LLMGenerator
from adaptive_biorag.schemas import RetrievalResult
from conftest import ScriptedLLM

EV = [RetrievalResult("D1", "D1::p0", "Valtrexin inhibits kernase-7 strongly.", "T", "s", 1.0, "x", 1)]


def test_invalid_and_valid_citations_marked():
    out = GenerationOutput("Claim [D1::p0] and [FAKE::p9].", citations=[
        {"passage_id": "D1::p0", "document_id": "D1"}, {"passage_id": "FAKE::p9", "document_id": "FAKE"},
        {"passage_id": "D1::p0", "document_id": "WRONG"}])
    v = validate_citations(out, EV)
    assert [c["valid"] for c in v.citations] == [True, False, False] and "FAKE::p9" in v.invalid_citations and v.limitations


def test_inline_extraction_and_autofill():
    assert extract_inline_citations("a [D1::p0, D2::p1] b [see note]") == ["D1::p0", "D2::p1"]
    v = validate_citations(GenerationOutput("Text [D1::p0]."), EV)
    assert v.citations[0]["valid"] and v.citations[0]["document_id"] == "D1"


def test_llm_generator_validates_unknown_citation_and_label():
    reply = json.dumps({"answer": "Valtrexin inhibits it [D1::p0][D9::p9].", "citations": [{"passage_id": "D9::p9", "document_id": "D9", "claim_supported": "x"}],
                        "limitations": [], "uncertainties": [], "insufficient_evidence": False, "label": "B"})
    out = LLMGenerator(ScriptedLLM([reply])).generate("q", EV, None, {"A": "x", "B": "y"})
    assert out.label == "B" and "D9::p9" in out.invalid_citations and out.generator == "llm:scripted"


def test_llm_generator_failure_modes_do_not_crash():
    for bad in ["garbage", LLMError("timeout")]:
        out = LLMGenerator(ScriptedLLM([bad])).generate("q", EV, None)
        assert out.insufficient_evidence and out.error


def test_extractive_cites_only_supplied_and_abstains():
    g = ExtractiveGenerator()
    out = g.generate("What does valtrexin inhibit?", EV, None)
    assert out.citations and all(c["valid"] for c in out.citations) and "[D1::p0]" in out.answer
    none = g.generate("Zebrafish fin regeneration QPL3?", EV, None)
    assert none.insufficient_evidence and none.citations == []
