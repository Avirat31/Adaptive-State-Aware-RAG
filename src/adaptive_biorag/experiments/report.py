"""Builds the results table ONLY from saved run outputs. Unavailable values are printed as n/a (never invented)."""
from __future__ import annotations

from pathlib import Path

from ..utils.serialization import read_json, write_csv

COLS = ["System", "Dataset", "Retrieval metric (doc recall@k / MRR)", "Answer metric", "Evidence support (cite validity / lexical proxy)",
        "Retrieval calls (mean)", "Latency s (mean)", "Estimated cost"]


def _f(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _kmax(dm: dict) -> int:
    return max(int(k.split('@')[1]) for k in dm if k.startswith('recall@'))


def row_for_run(run_dir: Path) -> dict:
    m, meta = read_json(run_dir / "metrics.json")["overall"], read_json(run_dir / "run_metadata.json")
    dm = m["retrieval"].get("doc_metrics") or {}
    a, c, e = m["answer"], m["citations"], m["efficiency"]
    if a.get("n_classification"):
        ans = f"accuracy={_f(a.get('cls_accuracy'))} macroF1={_f(a.get('cls_macro_f1'))} (n={a['n_classification']})"
    elif a.get("n_text"):
        ans = f"token-F1={_f(a.get('text_f1'))} contains={_f(a.get('text_contains'))} (n={a['n_text']})"
    else:
        ans = "n/a (no gold answers)"
    return {COLS[0]: meta.get("system_id") or run_dir.name, COLS[1]: meta["dataset"].get("dataset", "?") + f"/{meta['dataset'].get('split')}",
            COLS[2]: (f"recall@{_kmax(dm)}={_f(dm.get(f'recall@{_kmax(dm)}'))} / MRR={_f(dm.get('mrr'))} (n={m['retrieval']['n_with_gold_doc']})"
                      if dm else "n/a"),
            COLS[3]: ans, COLS[4]: f"{_f(c.get('citation_validity'))} / {_f(c.get('lexical_support_proxy'))}",
            COLS[5]: _f(e.get("retrieval_calls_mean"), 2), COLS[6]: _f(e.get("latency_total_s_mean"), 4), COLS[7]: _f(e.get("estimated_cost_sum"), 4)}


def build_report(run_dirs: list[Path], out_dir: Path) -> Path:
    rows = [row_for_run(Path(d)) for d in run_dirs]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "results_table.csv", rows)
    md = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    md += ["| " + " | ".join(str(r[c]) for c in COLS) + " |" for r in rows]
    md += ["", "_Generated from run outputs. Compare systems on identical examples/corpus/generator; lexical proxies are not "
           "human judgments; differences on small pilots are not statistically meaningful._"]
    p = out_dir / "results_table.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p
