from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import __version__
from ..config import config_hash
from ..corpus.index_manifest import build_corpus_index
from ..datasets import get_adapter, validate_examples
from ..evaluation.evaluator import evaluate_predictions, flatten_metrics
from ..generation.extractive import ExtractiveGenerator
from ..generation.llm_provider import LLMGenerator, OpenAICompatibleClient
from ..logging_utils import get_logger, setup_logging
from ..reasoning.decomposer import build_decomposer
from ..reasoning.orchestrator import Pipeline
from ..retrieval.factory import build_retriever
from ..retrieval.reranker import build_reranker
from ..routing.heuristic_router import build_router
from ..utils.reproducibility import environment_info, peak_rss_mb, set_seed
from ..utils.serialization import write_csv, write_json, write_jsonl

log = get_logger("runner")


def build_pipeline(cfg: dict, passages, index_dir, corpus_hash: str, llm_client=None) -> Pipeline:
    if llm_client is None and (cfg["generation"]["provider"] == "llm" or cfg["decomposition"]["method"] == "llm"):
        llm_client = OpenAICompatibleClient(cfg)
    g = cfg["generation"]
    generator = (LLMGenerator(llm_client) if g["provider"] == "llm" else ExtractiveGenerator(g["max_sentences"], g["min_overlap"]))
    return Pipeline(cfg, build_retriever(cfg, passages, index_dir, corpus_hash), build_router(cfg),
                    build_decomposer(cfg, llm_client), generator, build_reranker(cfg), llm_client)


def make_run_dir(cfg: dict) -> Path:
    root = Path(cfg["project"]["output_dir"])
    name = cfg["project"].get("run_name") or "run"
    date = datetime.now().strftime("%Y-%m-%d")
    for n in range(1, 10000):
        d = root / f"{date}_{name}_{n:03d}"
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a run directory")


def _progress(it, total, desc):
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=desc)
    except ImportError:
        return it


def summarize(cfg: dict, res: dict, n: int) -> str:
    o = res["overall"]
    lines = [f"AdaptiveBioRAG run summary", f"dataset={cfg['dataset']['name']} split={cfg['dataset']['split']} n={n}",
             f"retrieval={cfg['retrieval']['method']} reranker={cfg['retrieval']['use_reranker']} dense_backend={cfg['retrieval']['dense_backend']}",
             f"routing_enabled={cfg['routing']['enabled']} (forced={cfg['routing']['forced_route']}) memory={cfg['memory']['enabled']}",
             f"generator={cfg['generation']['provider']} decomposer={cfg['decomposition']['method']}",
             f"routes={o['routing']['route_counts']} mean_iterations={o['routing']['mean_iterations']}",
             f"stop_reasons={o['routing']['stop_reasons']}"]
    dm = o["retrieval"].get("doc_metrics")
    lines.append("doc-level retrieval (n=%s with gold): %s" % (o["retrieval"]["n_with_gold_doc"],
                 {k: round(v, 3) for k, v in dm.items()} if dm else "unavailable (no gold documents)"))
    lines.append(f"answer metrics: { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o['answer'].items()} }")
    lines.append(f"citations: { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o['citations'].items()} }")
    lines.append(f"errors in {res['n_errors']} question(s) (see errors.jsonl)")
    lines.append("NOTE: numbers describe this run only; they are not evidence of improvement without the comparisons "
                 "described in the README. Lexical proxies are not human judgments.")
    return "\n".join(lines)


def run_experiment(cfg: dict, quiet: bool = False, config_source: str | None = None) -> dict:
    run_dir = make_run_dir(cfg)
    setup_logging(cfg["logging"]["level"], run_dir / "run.log")
    set_seed(cfg["project"]["seed"])
    t_start = time.time()
    started = datetime.now(timezone.utc).isoformat()
    try:
        adapter = get_adapter(cfg)
        examples = adapter.load()
        vrep = validate_examples(examples)
        passages, manifest, index_dir = build_corpus_index(cfg)
        pipe = build_pipeline(cfg, passages, index_dir, manifest["corpus_hash"])
    except Exception as e:
        write_json(run_dir / "errors.json", {"stage": "setup", "error": f"{type(e).__name__}: {e}"})
        log.error("Setup failed: %s", e)
        raise
    log.info("run=%s n=%d passages=%d", run_dir.name, len(examples), len(passages))
    preds, errors = [], []
    it = examples if quiet else _progress(examples, len(examples), run_dir.name)
    for ex in it:
        try:
            preds.append(pipe.answer(ex, ex.metadata.get("options")))
            errors += [{"question_id": ex.question_id, **e} for e in preds[-1]["errors"]]
        except Exception as e:                          # noqa: BLE001 - one bad question must not kill the run
            errors.append({"question_id": ex.question_id, "stage": "pipeline", "error": f"{type(e).__name__}: {e}"})
            log.error("question %s failed: %s", ex.question_id, e)
    res = evaluate_predictions(preds, cfg["evaluation"]["retrieval_k"]) if preds else {"overall": {}, "subsets": {}, "n_errors": len(errors)}
    # --- artifacts -------------------------------------------------------------------------------------------------
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if cfg["evaluation"]["save_predictions"]:
        write_jsonl(run_dir / "predictions.jsonl", preds)
    write_jsonl(run_dir / "retrieval_results.jsonl", ({"question_id": p["question_id"], "route": p["route"],
                "ranking": p["retrieval_ranking"]} for p in preds))
    write_jsonl(run_dir / "iteration_logs.jsonl", ({"question_id": p["question_id"], "route": p["route"], "routing": p["routing"],
                "sub_questions": p["sub_questions"], "stop_reason": p["stop_reason"], "iterations": p["iteration_logs"]} for p in preds))
    write_jsonl(run_dir / "errors.jsonl", errors)
    if pipe.llm is not None:
        write_jsonl(run_dir / "llm_calls.jsonl", pipe.llm.calls)       # prompts + token counts; never credentials
    if preds:
        write_json(run_dir / "metrics.json", res)
        write_csv(run_dir / "metrics.csv", flatten_metrics(res))
        rnd = random.Random(cfg["project"]["seed"])
        sample = rnd.sample(preds, min(cfg["evaluation"]["inspection_sample"], len(preds)))
        write_jsonl(run_dir / "inspection_sample.jsonl", ({"question_id": p["question_id"], "question": p["question"],
                    "route": p["route"], "answer": p["answer"], "evidence_ids": [e["passage_id"] for e in p["final_evidence"]],
                    "manual_label_supported": None, "manual_notes": ""} for p in sample))
    uses_ext = cfg["generation"]["provider"] == "llm" or cfg["decomposition"]["method"] == "llm"
    write_json(run_dir / "run_metadata.json", {
        "run_id": run_dir.name, "system_id": cfg["project"].get("run_name"), "package_version": __version__,
        "config_hash": config_hash(cfg), "config_source": config_source, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": round(time.time() - t_start, 3),
        "seed": cfg["project"]["seed"], "environment": environment_info(), "peak_rss_mb": peak_rss_mb(),
        "dataset": adapter.info, "dataset_validation": vrep, "corpus_manifest": {k: v for k, v in manifest.items() if k != "load_failures"},
        "corpus_load_failures": len(manifest.get("load_failures", [])),
        "models": {"dense_backend": cfg["retrieval"]["dense_backend"], "dense_model": cfg["retrieval"]["dense_model"]
                   if cfg["retrieval"]["dense_backend"] != "hashing" else f"hashing(dim={cfg['retrieval']['dense_dim']})",
                   "reranker": (cfg["retrieval"]["reranker_method"] if cfg["retrieval"]["use_reranker"] else None),
                   "llm_model": cfg["llm"]["model"] if uses_ext else None},
        "external_api_calls_enabled": uses_ext, "deterministic_components_only": not uses_ext})
    (run_dir / "summary.txt").write_text(summarize(cfg, res, len(preds)) if preds else "No predictions produced.", encoding="utf-8")
    log.info("Run complete: %s", run_dir)
    return {"run_dir": run_dir, "metrics": res, "n_predictions": len(preds), "n_errors": len(errors)}
