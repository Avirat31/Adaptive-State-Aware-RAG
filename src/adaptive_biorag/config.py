"""YAML configuration with inheritance (`base:`), strict unknown-key detection and cross-field validation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "project": {"name": "AdaptiveBioRAG", "run_name": None, "seed": 42, "output_dir": "outputs"},
    "dataset": {
        "name": "synthetic",          # synthetic | jsonl | pubmedqa | bioasq | mirage | medhopqa
        "split": "test",
        "data_dir": "examples",
        "path": None,                 # explicit file; otherwise resolved from data_dir by the adapter
        "subset": None,               # MIRAGE subset key (e.g. "pubmedqa") / other adapter-specific selector
        "limit": None,                # pilot runs: first N examples (deterministic order)
        "split_file": None,           # PubMedQA fold / id-list file
        "field_map": {},              # MedHopQA/jsonl: {"question": "...", "answer": "...", "id": "...", "gold_docs": "..."}
        "provide_context": False,     # if True, adapters may expose dataset-supplied context to the pipeline (NOT retrieval QA)
    },
    "corpus": {
        "source": "jsonl",            # jsonl | pubmedqa_contexts | bioasq_snippets
        "path": "examples/sample_documents.jsonl",
        "passage_size": 80,           # words per passage
        "passage_overlap": 20,        # words of overlap between consecutive passages
        "min_passage_words": 3,
        "index_dir": "indexes/default",
    },
    "retrieval": {
        "method": "hybrid",           # dense | bm25 | hybrid
        "top_k_sparse": 50, "top_k_dense": 50, "top_k_fused": 10, "rrf_k0": 60,
        "dense_backend": "hashing",   # hashing (offline, deterministic stand-in) | sentence_transformers
        "dense_model": "BAAI/bge-m3", # used only by sentence_transformers backend
        "dense_dim": 512,             # hashing backend only
        "query_prefix": "", "passage_prefix": "",
        "dense_batch_size": 32,
        "use_reranker": False, "reranker_method": "lexical",  # lexical (offline) | cross_encoder
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "rerank_candidates": 20,
        "final_evidence_k": 5,
        "use_graph": False,           # reserved; graph retrieval is NOT implemented in this release
    },
    "routing": {
        "enabled": True, "method": "heuristic", "complexity_threshold": 0.4,
        "forced_route": "single_step",   # used when routing.enabled is false
        "learned_model_path": None,
        "weights": {"extra_clause": 0.25, "extra_wh": 0.20, "comparison": 0.30, "relation": 0.15,
                    "multi_entity": 0.15, "chain": 0.20, "long_question": 0.15, "multi_question_mark": 0.30},
    },
    "decomposition": {"method": "deterministic", "max_sub_questions": 4},
    "memory": {"enabled": True, "deduplicate": True, "detect_conflicts": False, "subq_support_threshold": 0.6},
    "stopping": {"max_iterations": 3, "minimum_new_evidence": 1, "evidence_coverage_threshold": 1.0,
                 "max_duplicate_ratio": 0.9, "min_evidence_passages": 1},
    "generation": {"provider": "extractive", "max_sentences": 3, "min_overlap": 0.3, "temperature": 0.0,
                   "max_output_tokens": 512},
    "llm": {"model": "gpt-4o-mini", "base_url": None, "api_key_env": "ADAPTIVE_BIORAG_API_KEY",
            "timeout_seconds": 60, "max_retries": 2, "temperature": 0.0, "max_output_tokens": 512,
            "price_per_1k_input": None, "price_per_1k_output": None},   # prices stay None unless YOU set them
    "evaluation": {"retrieval_k": [1, 5, 10], "save_predictions": True, "inspection_sample": 10},
    "logging": {"level": "INFO"},
}
FREEFORM_PATHS = {"dataset.field_map"}
_ENUMS = {
    ("dataset", "name"): {"synthetic", "jsonl", "pubmedqa", "bioasq", "mirage", "medhopqa"},
    ("corpus", "source"): {"jsonl", "pubmedqa_contexts", "bioasq_snippets"},
    ("retrieval", "method"): {"dense", "bm25", "hybrid"},
    ("retrieval", "dense_backend"): {"hashing", "sentence_transformers"},
    ("retrieval", "reranker_method"): {"lexical", "cross_encoder"},
    ("routing", "method"): {"heuristic", "learned"},
    ("routing", "forced_route"): {"single_step", "multi_step"},
    ("decomposition", "method"): {"deterministic", "llm"},
    ("generation", "provider"): {"extractive", "llm"},
}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_unknown(cfg: dict, schema: dict, path: str = "") -> None:
    for k, v in cfg.items():
        full = f"{path}{k}"
        if k not in schema:
            raise ConfigError(f"Unknown configuration key '{full}'. Valid keys here: {sorted(schema)}")
        if isinstance(schema[k], dict) and schema[k] and full not in FREEFORM_PATHS:
            if not isinstance(v, dict):
                raise ConfigError(f"'{full}' must be a mapping")
            _check_unknown(v, schema[k], full + ".")


def _load_raw(path: Path, _seen: tuple = ()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ConfigError(f"Circular 'base' inheritance at {path}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    bases = raw.pop("base", None)
    merged: dict = {}
    for b in ([bases] if isinstance(bases, str) else (bases or [])):
        merged = deep_merge(merged, _load_raw(path.parent / b, _seen + (path,)))
    return deep_merge(merged, raw)


def parse_overrides(items: list[str] | None) -> dict:
    """['retrieval.method=bm25', 'stopping.max_iterations=2'] -> nested dict (values parsed as YAML scalars)."""
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ConfigError(f"Override '{it}' must look like section.key=value")
        key, val = it.split("=", 1)
        cur = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    raw = _load_raw(Path(path)) if path else {}
    raw = deep_merge(raw, overrides or {})
    _check_unknown(raw, DEFAULTS)
    cfg = deep_merge(DEFAULTS, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for (sec, key), allowed in _ENUMS.items():
        if cfg[sec][key] not in allowed:
            raise ConfigError(f"{sec}.{key}='{cfg[sec][key]}' invalid; choose one of {sorted(allowed)}")
    c, r, s, m = cfg["corpus"], cfg["retrieval"], cfg["stopping"], cfg["memory"]
    if c["passage_size"] < 5:
        raise ConfigError("corpus.passage_size must be >= 5 words")
    if not 0 <= c["passage_overlap"] < c["passage_size"]:
        raise ConfigError("corpus.passage_overlap must satisfy 0 <= overlap < passage_size")
    for k in ("top_k_sparse", "top_k_dense", "top_k_fused", "rrf_k0", "final_evidence_k", "rerank_candidates"):
        if not isinstance(r[k], int) or r[k] < 1:
            raise ConfigError(f"retrieval.{k} must be a positive integer")
    if r["final_evidence_k"] > r["top_k_fused"]:
        raise ConfigError("retrieval.final_evidence_k cannot exceed retrieval.top_k_fused")
    if r["use_graph"]:
        raise ConfigError("retrieval.use_graph=true: graph retrieval is an optional extension and is NOT implemented "
                          "in this release (see README 'Optional / not implemented').")
    if r["use_reranker"] and r["rerank_candidates"] < r["final_evidence_k"]:
        raise ConfigError("retrieval.rerank_candidates must be >= final_evidence_k when the reranker is enabled")
    if not 1 <= s["max_iterations"] <= 10:
        raise ConfigError("stopping.max_iterations must be in [1, 10] (bounded loop)")
    if not 0 < s["evidence_coverage_threshold"] <= 1:
        raise ConfigError("stopping.evidence_coverage_threshold must be in (0, 1]")
    if not 0 < m["subq_support_threshold"] <= 1:
        raise ConfigError("memory.subq_support_threshold must be in (0, 1]")
    if not 0 <= cfg["routing"]["complexity_threshold"] <= 1:
        raise ConfigError("routing.complexity_threshold must be in [0, 1]")
    if not 1 <= cfg["decomposition"]["max_sub_questions"] <= 8:
        raise ConfigError("decomposition.max_sub_questions must be in [1, 8]")
    if cfg["routing"]["method"] == "learned" and not cfg["routing"]["learned_model_path"]:
        raise ConfigError("routing.method=learned requires routing.learned_model_path")
    if any((not isinstance(k, int)) or k < 1 for k in cfg["evaluation"]["retrieval_k"]):
        raise ConfigError("evaluation.retrieval_k must be a list of positive integers")
    if cfg["dataset"]["limit"] is not None and cfg["dataset"]["limit"] < 1:
        raise ConfigError("dataset.limit must be null or >= 1")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
