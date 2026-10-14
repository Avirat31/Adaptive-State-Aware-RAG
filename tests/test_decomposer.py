import json

from adaptive_biorag.reasoning.decomposer import DeterministicDecomposer, LLMDecomposer, bridge_terms
from adaptive_biorag.generation.llm_provider import LLMError
from conftest import ScriptedLLM

Q = "Which gene is associated with Hollenbeck syndrome, and what cellular process does its protein product regulate?"


def test_deterministic_splits_compound_and_bounds():
    d = DeterministicDecomposer(4)
    subs = d.decompose(Q)
    assert len(subs) == 2 and "Hollenbeck" in subs[0] and subs[1].lower().startswith("what")
    assert len(DeterministicDecomposer(1).decompose(Q)) == 1


def test_deterministic_conservative_fallback():
    assert DeterministicDecomposer().decompose("What does KRN7 encode?") == ["What does KRN7 encode?"]
    assert len(DeterministicDecomposer(3).decompose("   ")) <= 3


def test_llm_valid_json_and_grounding_filter():
    reply = json.dumps({"sub_questions": ["Which gene is linked to Hollenbeck syndrome?", "What does the Zorblax protein do?",
                                          "Which gene is linked to Hollenbeck syndrome?", 5]})
    d = LLMDecomposer(ScriptedLLM([reply]), 4)
    out = d.decompose(Q)
    assert out == ["Which gene is linked to Hollenbeck syndrome?"] and d.last_info["fallback"] is False and len(d.last_info["dropped"]) == 2


def test_llm_fallbacks_on_malformed_or_failure():
    for bad in ["not json at all", json.dumps({"sub_questions": "x"}), json.dumps({"sub_questions": []}), LLMError("boom")]:
        d = LLMDecomposer(ScriptedLLM([bad]), 4)
        out = d.decompose(Q)
        assert d.last_info["fallback"] is True and len(out) == 2


def test_llm_handles_code_fences():
    d = LLMDecomposer(ScriptedLLM(["```json\n{\"sub_questions\": [\"What regulates the Hollenbeck syndrome gene?\"]}\n```"]), 3)
    assert d.decompose(Q) == ["What regulates the Hollenbeck syndrome gene?"]


def test_bridge_terms_and_follow_up():
    t = bridge_terms([("KRN7 protein", "KRN7 encodes kernase-7 and phosphorylates CDX2.")], {"gene"})
    assert "KRN7" in t and "CDX2" in t
    q = DeterministicDecomposer().follow_up_query("Which drug inhibits it?", "orig", [("T", "Valtrexin inhibits kernase-7 IC50")])
    assert q.startswith("Which drug inhibits it?") and "IC50" in q
