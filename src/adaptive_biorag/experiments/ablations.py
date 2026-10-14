from __future__ import annotations

from ..config import ConfigError, deep_merge, validate_config
from .baselines import apply_preset
from .runner import run_experiment


def run_suite(cfg: dict, systems: list[str], quiet: bool = True) -> list[dict]:
    out = []
    for s in systems:
        out.append({"system": s, **run_experiment(apply_preset(cfg, s), quiet=quiet)})
    return out


def sweep(cfg: dict, section: str, key: str, values: list, allow_test: bool = False, quiet: bool = True) -> list[dict]:
    """One-parameter sweep (e.g. stopping.max_iterations for RQ6, routing.complexity_threshold for the router).
    Refuses to run on the test split unless explicitly allowed, to prevent tuning on test data."""
    if cfg["dataset"]["split"] == "test" and not allow_test:
        raise ConfigError("Refusing to sweep on split 'test' (data-leakage guard). Use a validation split, or pass "
                          "--allow-test ONLY for synthetic/demo data.")
    out = []
    for v in values:
        c = deep_merge(cfg, {section: {key: v}})
        c["project"]["run_name"] = f"sweep_{section}_{key}_{v}"
        validate_config(c)
        out.append({"value": v, **run_experiment(c, quiet=quiet)})
    return out
