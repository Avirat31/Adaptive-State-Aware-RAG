import json

import pytest

from adaptive_biorag.cli import main
from adaptive_biorag.config import ConfigError
from adaptive_biorag.experiments.ablations import run_suite, sweep
from adaptive_biorag.experiments.runner import run_experiment
from adaptive_biorag.utils.serialization import read_json, read_jsonl
from conftest import ROOT

ARTIFACTS = ["config.yaml", "run_metadata.json", "predictions.jsonl", "retrieval_results.jsonl", "iteration_logs.jsonl",
             "metrics.json", "metrics.csv", "errors.jsonl", "summary.txt", "inspection_sample.jsonl"]


def test_adaptive_run_end_to_end(cfg):
    cfg["project"]["run_name"] = "e2e"
    r = run_experiment(cfg, quiet=True)
    d = r["run_dir"]
    assert all((d / a).exists() for a in ARTIFACTS) and r["n_predictions"] == 9 and r["n_errors"] == 0
    preds = {p["question_id"]: p for p in read_jsonl(d / "predictions.jsonl")}
    assert preds["S001"]["route"] == "single_step" and preds["S006"]["route"] == "multi_step"
    assert preds["S001"]["n_iterations"] == 1 and 1 <= preds["S006"]["n_iterations"] <= 3
    assert all(c["valid"] for p in preds.values() for c in p["answer"]["citations"])
    meta = read_json(d / "run_metadata.json")
    assert meta["external_api_calls_enabled"] is False and "api_key" not in json.dumps(meta).lower()
    m = read_json(d / "metrics.json")["overall"]
    assert m["retrieval"]["n_missing_gold_doc"] == 1 and m["retrieval"]["doc_metrics"]["recall@10"] > 0


def test_gold_never_reaches_generator(cfg, monkeypatch):
    seen = []
    from adaptive_biorag.generation.extractive import ExtractiveGenerator
    orig = ExtractiveGenerator.generate
    def spy(self, question, evidence, state, options=None):
        seen.append((question, [e.metadata for e in evidence])); return orig(self, question, evidence, state, options)
    monkeypatch.setattr(ExtractiveGenerator, "generate", spy)
    run_experiment(cfg, quiet=True)
    assert "gold" not in json.dumps(seen).lower() and len(seen) == 9


def test_runs_do_not_overwrite(cfg):
    a, b = run_experiment(cfg, quiet=True)["run_dir"], run_experiment(cfg, quiet=True)["run_dir"]
    assert a != b and a.exists() and b.exists()


def test_suite_distinct_behaviour(cfg):
    res = {r["system"]: r["metrics"]["overall"]["routing"] for r in run_suite(cfg, ["B0", "B2", "B4", "B5", "B6"])}
    assert res["B0"]["route_counts"] == {"single_step": 9} and res["B2"]["route_counts"] == {"single_step": 9}
    assert res["B4"]["route_counts"] == {"multi_step": 9} and res["B4"]["mean_iterations"] > 1
    assert set(res["B5"]["route_counts"]) == {"single_step", "multi_step"}
    assert "fixed_iterations_complete" in res["B4"]["stop_reasons"] or "no_new_queries" in res["B4"]["stop_reasons"]
    assert "single_step_complete" in res["B6"]["stop_reasons"]


def test_sweep_blocks_test_split(cfg):
    with pytest.raises(ConfigError, match="leakage"):
        sweep(cfg, "stopping", "max_iterations", [1, 2])
    out = sweep(cfg, "stopping", "max_iterations", [1, 2], allow_test=True)
    assert out[0]["metrics"]["overall"]["routing"]["mean_iterations"] <= 1.0001 or out[0]["n_predictions"] == 9


def test_reranker_and_dense_paths_run(cfg):
    for m in ("dense", "bm25"):
        cfg["retrieval"].update(method=m, use_reranker=True)
        assert run_experiment(cfg, quiet=True)["n_predictions"] == 9


def test_cli_exit_codes_and_evaluate(tmp_path, capsys):
    base = ["--config", str(ROOT / "configs/default.yaml"), "--output-dir", str(tmp_path / "o"), "--set", f"corpus.index_dir={tmp_path/'i'}"]
    assert main(["run", "--quiet", *base]) == 0
    pf = next((tmp_path / "o").glob("*/predictions.jsonl"))
    assert main(["evaluate", "--predictions", str(pf)]) == 0 and (pf.parent / "metrics_reeval.json").exists()
    assert main(["route", "--question", "Compare A versus B; how do they differ?", *base]) == 0
    assert main(["run", "--config", str(ROOT / "configs/default.yaml"), "--set", "retrieval.method=nope"]) == 2
    assert main(["run", "--config", str(ROOT / "configs/default.yaml"), "--set", "dataset.path=/no/such.jsonl", "--set", "dataset.name=jsonl",
                 "--output-dir", str(tmp_path / "o")]) == 2
    assert main(["inspect-dataset", *base]) == 0
    assert "ERROR" in capsys.readouterr().err


def test_invalid_question_recorded_not_fatal(cfg):
    from adaptive_biorag.corpus.index_manifest import build_corpus_index
    from adaptive_biorag.experiments.runner import build_pipeline
    from adaptive_biorag.schemas import Example
    ps, man, idx = build_corpus_index(cfg)
    with pytest.raises(ValueError, match="Invalid question"):
        build_pipeline(cfg, ps, idx, man["corpus_hash"]).answer(Example("x", "", "d", "s"))
