"""Command line interface:  python -m adaptive_biorag.cli <command> --config configs/default.yaml [--set key=value ...]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config, parse_overrides
from .datasets import DatasetError
from .utils.serialization import dumps, read_jsonl, write_csv, write_json


class StageError(RuntimeError):
    def __init__(self, stage: str, err: Exception):
        super().__init__(f"[{stage}] {type(err).__name__}: {err}")


def _cfg(args) -> dict:
    ov = parse_overrides(args.set)
    if getattr(args, "limit", None):
        ov.setdefault("dataset", {})["limit"] = args.limit
    if getattr(args, "output_dir", None):
        ov.setdefault("project", {})["output_dir"] = args.output_dir
    try:
        return load_config(args.config, ov)
    except ConfigError as e:
        raise StageError("config", e) from e


def cmd_inspect(args):
    from .datasets import get_adapter, validate_examples
    cfg = _cfg(args)
    try:
        a = get_adapter(cfg)
        ex = a.load()
        rep = validate_examples(ex, strict=False)
    except DatasetError as e:
        raise StageError("dataset", e) from e
    print(dumps({"dataset_info": a.info, "validation": rep, "first_example": ex[0].to_dict() if ex else None}, indent=2))
    return 1 if rep["errors"] else 0


def cmd_build_index(args):
    from .corpus.index_manifest import build_corpus_index
    from .retrieval.factory import build_retriever
    cfg = _cfg(args)
    try:
        passages, man, idx = build_corpus_index(cfg, force=args.force)
        build_retriever({**cfg, "retrieval": {**cfg["retrieval"], "method": "hybrid"}}, passages, idx, man["corpus_hash"])  # also caches embeddings
    except Exception as e:
        raise StageError("build-index", e) from e
    print(dumps({"index_dir": str(idx), **{k: man[k] for k in ("n_documents", "n_passages", "corpus_hash", "n_unmapped_documents")},
                 "load_failures": len(man["load_failures"]), "dedup": man["dedup"]}, indent=2))
    return 0


def cmd_run(args):
    from .experiments.runner import run_experiment
    cfg = _cfg(args)
    try:
        r = run_experiment(cfg, quiet=args.quiet, config_source=str(args.config))
    except Exception as e:
        raise StageError("run", e) from e
    print(f"\nRun directory: {r['run_dir']}\n" + (r["run_dir"] / "summary.txt").read_text(encoding="utf-8"))
    return 0 if r["n_predictions"] else 1


def cmd_suite(args):
    from .experiments.ablations import run_suite
    from .experiments.report import build_report
    cfg = _cfg(args)
    try:
        res = run_suite(cfg, args.systems)
        p = build_report([r["run_dir"] for r in res], Path(cfg["project"]["output_dir"]) / "reports")
    except Exception as e:
        raise StageError("run-suite", e) from e
    print(p.read_text(encoding="utf-8"))
    return 0


def cmd_sweep(args):
    from .experiments.ablations import sweep
    from .experiments.report import build_report
    import yaml
    cfg = _cfg(args)
    sec, key = args.param.split(".", 1)
    vals = [yaml.safe_load(v) for v in args.values]
    try:
        res = sweep(cfg, sec, key, vals, allow_test=args.allow_test)
        p = build_report([r["run_dir"] for r in res], Path(cfg["project"]["output_dir"]) / "reports")
    except Exception as e:
        raise StageError("sweep", e) from e
    print(p.read_text(encoding="utf-8"))
    return 0


def cmd_evaluate(args):
    from .evaluation.evaluator import evaluate_predictions, flatten_metrics
    cfg = load_config(args.config) if args.config else load_config(None)
    try:
        preds = read_jsonl(Path(args.predictions))
        res = evaluate_predictions(preds, cfg["evaluation"]["retrieval_k"])
    except Exception as e:
        raise StageError("evaluate", e) from e
    out = Path(args.predictions).with_name("metrics_reeval.json")
    write_json(out, res)
    write_csv(out.with_suffix(".csv"), flatten_metrics(res))
    print(dumps(res["overall"]["retrieval"], indent=2), "\n", dumps(res["overall"]["answer"], indent=2), f"\nSaved {out}")
    return 0


def cmd_report(args):
    from .experiments.report import build_report
    p = build_report([Path(d) for d in args.runs], Path(args.out))
    print(p.read_text(encoding="utf-8"))
    return 0


def cmd_route(args):
    from .routing.heuristic_router import build_router
    cfg = _cfg(args)
    print(dumps(build_router({**cfg, "routing": {**cfg["routing"], "enabled": True}}).route(args.question), indent=2))
    return 0


def cmd_ask(args):
    from .corpus.index_manifest import build_corpus_index
    from .experiments.runner import build_pipeline
    from .schemas import Example
    cfg = _cfg(args)
    try:
        passages, man, idx = build_corpus_index(cfg)
        pipe = build_pipeline(cfg, passages, idx, man["corpus_hash"])
        p = pipe.answer(Example("adhoc", args.question, "adhoc", "n/a"))
    except Exception as e:
        raise StageError("ask", e) from e
    if args.json:
        print(dumps(p, indent=2))
        return 0
    print(f"ROUTE: {p['route']} (score={p['routing']['complexity_score']}, reasons={p['routing']['reason_codes']})")
    print(f"SUB-QUESTIONS: {p['sub_questions']}")
    for l in p["iteration_logs"]:
        print(f"  iter {l['iteration']}: queries={l['queries']} new={l['new_evidence_ids']} dup={l['duplicates']} "
              f"cov {l['coverage_before']:.2f}->{l['coverage_after']:.2f} stop={l['stop_decision']['reason']}")
    print("EVIDENCE:", [e["passage_id"] for e in p["final_evidence"]])
    a = p["answer"]
    print(f"ANSWER: {a['answer']}\nUNCERTAINTIES: {a['uncertainties']}\nLIMITATIONS: {a['limitations']}")
    print("(Research prototype: literature QA only, not clinical advice.)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="adaptive_biorag", description="AdaptiveBioRAG research toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, config=True):
        if config:
            p.add_argument("--config", default="configs/default.yaml", help="YAML config (supports `base:` inheritance)")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. retrieval.method=bm25")
        p.add_argument("--limit", type=int, help="use only the first N examples (pilot)")
        p.add_argument("--output-dir", help="override project.output_dir")

    p = sub.add_parser("inspect-dataset", help="load, validate and report dataset statistics"); common(p); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("build-index", help="chunk corpus, deduplicate, cache embeddings"); common(p)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_build_index)
    p = sub.add_parser("run", help="run one experiment configuration"); common(p)
    p.add_argument("--quiet", action="store_true"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("run-suite", help="run baseline presets B0..B6 and build the results table"); common(p)
    p.add_argument("--systems", nargs="+", default=["B0", "B1", "B2", "B3", "B4", "B5", "B6"]); p.set_defaults(fn=cmd_suite)
    p = sub.add_parser("sweep", help="one-parameter sweep, e.g. --param stopping.max_iterations --values 1 2 3"); common(p)
    p.add_argument("--param", required=True); p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--allow-test", action="store_true", help="demo only: permit sweeping on split 'test'"); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("evaluate", help="(re)compute metrics from predictions.jsonl")
    p.add_argument("--predictions", required=True); p.add_argument("--config"); p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("report", help="build results table from run directories")
    p.add_argument("runs", nargs="+"); p.add_argument("--out", default="outputs/reports"); p.set_defaults(fn=cmd_report)
    p = sub.add_parser("route", help="show the routing decision for a question"); common(p)
    p.add_argument("--question", required=True); p.set_defaults(fn=cmd_route)
    p = sub.add_parser("ask", help="answer one ad-hoc question over the configured corpus"); common(p)
    p.add_argument("--question", required=True); p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_ask)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except StageError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
