"""Baseline presets B0..B7. Each preset really switches components off (different code paths, not relabelled runs).
The configs/*.yaml files mirror these presets (checked by tests/test_config.py)."""
from __future__ import annotations

from ..config import ConfigError, deep_merge, validate_config

_OFF = {"routing": {"enabled": False, "forced_route": "single_step"}, "memory": {"enabled": False}}
PRESETS: dict[str, dict] = {
    "B0": {"description": "Dense retrieval + generation", **deep_merge(_OFF, {"retrieval": {"method": "dense", "use_reranker": False}})},
    "B1": {"description": "BM25 + generation", **deep_merge(_OFF, {"retrieval": {"method": "bm25", "use_reranker": False}})},
    "B2": {"description": "Hybrid BM25+dense (RRF)", **deep_merge(_OFF, {"retrieval": {"method": "hybrid", "use_reranker": False}})},
    "B3": {"description": "Hybrid + reranking", **deep_merge(_OFF, {"retrieval": {"method": "hybrid", "use_reranker": True}})},
    "B4": {"description": "Hybrid + fixed multi-step (no memory/stopping)",
           "retrieval": {"method": "hybrid", "use_reranker": False},
           "routing": {"enabled": False, "forced_route": "multi_step"}, "memory": {"enabled": False}},
    "B5": {"description": "Adaptive routing (multi-step branch without memory/stopping)",
           "retrieval": {"method": "hybrid", "use_reranker": False}, "routing": {"enabled": True}, "memory": {"enabled": False}},
    "B6": {"description": "Adaptive + evidence-aware memory and stopping",
           "retrieval": {"method": "hybrid", "use_reranker": False}, "routing": {"enabled": True}, "memory": {"enabled": True}},
    "B7": {"description": "B6 + graph retrieval (NOT implemented)", "retrieval": {"method": "hybrid", "use_graph": True},
           "routing": {"enabled": True}, "memory": {"enabled": True}},
}


def apply_preset(cfg: dict, system_id: str) -> dict:
    if system_id not in PRESETS:
        raise ConfigError(f"Unknown system '{system_id}'. Choose from {sorted(PRESETS)}")
    over = {k: v for k, v in PRESETS[system_id].items() if k != "description"}
    out = deep_merge(cfg, over)
    out["project"]["run_name"] = system_id
    validate_config(out)          # B7 fails here with an explicit 'graph not implemented' error
    return out
