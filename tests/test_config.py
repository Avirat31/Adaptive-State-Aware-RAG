import pytest

from adaptive_biorag.config import ConfigError, load_config, parse_overrides
from adaptive_biorag.experiments.baselines import PRESETS, apply_preset
from conftest import ROOT


def test_default_loads_and_validates():
    c = load_config(ROOT / "configs/default.yaml")
    assert c["retrieval"]["method"] == "hybrid" and c["stopping"]["max_iterations"] == 3


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("retrieval: {metod: bm25}\n")
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        load_config(p)


@pytest.mark.parametrize("over", [{"retrieval": {"method": "faiss"}}, {"corpus": {"passage_overlap": 999}},
                                  {"stopping": {"max_iterations": 0}}, {"stopping": {"max_iterations": 99}},
                                  {"retrieval": {"final_evidence_k": 50}}, {"routing": {"complexity_threshold": 2}},
                                  {"routing": {"method": "learned"}}, {"retrieval": {"use_graph": True}}])
def test_invalid_values_rejected(over):
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs/default.yaml", over)


def test_inheritance_and_cli_overrides():
    c = load_config(ROOT / "configs/baseline_bm25.yaml", parse_overrides(["stopping.max_iterations=2"]))
    assert c["retrieval"]["method"] == "bm25" and c["stopping"]["max_iterations"] == 2 and c["routing"]["enabled"] is False


def test_missing_file_message():
    with pytest.raises(ConfigError, match="not found"):
        load_config("nope.yaml")


@pytest.mark.parametrize("yaml_name,sid", [("baseline_dense", "B0"), ("baseline_bm25", "B1"), ("baseline_hybrid", "B2"),
                                           ("baseline_reranked", "B3"), ("multistep", "B4"), ("adaptive", "B5"),
                                           ("adaptive_memory", "B6")])
def test_yaml_matches_presets(yaml_name, sid):
    y, p = load_config(ROOT / f"configs/{yaml_name}.yaml"), apply_preset(load_config(ROOT / "configs/default.yaml"), sid)
    for sec in ("retrieval", "routing", "memory"):
        assert y[sec] == p[sec], (sid, sec)


def test_b7_graph_not_implemented():
    with pytest.raises(ConfigError, match="NOT implemented"):
        apply_preset(load_config(ROOT / "configs/default.yaml"), "B7")
    assert set(PRESETS) == {f"B{i}" for i in range(8)}
