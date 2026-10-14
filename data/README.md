# Data preparation

Nothing is downloaded automatically. Raw data is never committed (see `.gitignore`). Keep three layers apart:
**downloaded** (`data/raw/`, yours), **processed** (`data/processed/`, optional), **generated indexes** (`corpus.index_dir`, e.g. `indexes/...`).

> **Verification status.** The MIRAGE repository was confirmed (single `benchmark.json`, 7,663 questions, five subsets, ground-truth
> snippets removed). The exact per-record field names of MIRAGE, BioASQ, PubMedQA and MedHopQA were **not** inspected from the live
> releases when this code was written. Each adapter validates the schema it expects and fails with a clear message; run
> `inspect-dataset` on your download first and adjust the adapter / `dataset.field_map` if the schema differs. Subset sizes
> quoted in the project brief (1,089 / 1,273 / 4,183 / 500 / 618) must be checked with `inspect-dataset` before being cited.

| Dataset | Files you provide | Config | Gold evidence | Notes |
|---|---|---|---|---|
| synthetic | `examples/*.jsonl` (shipped, fictional entities) | `dataset.name: synthetic` | doc IDs | smoke tests only |
| PubMedQA | `ori_pqal.json` or a fold file (`QUESTION`, `CONTEXTS`, `final_decision`) | `dataset: {name: pubmedqa, path: ..., split_file: ...}`, `corpus.source: pubmedqa_contexts` | document = PMID | retrieval-QA variant (context withheld). The closed-pool corpus contains each question's own abstract, so retrieval is optimistic. |
| BioASQ | Task-b JSON (`questions[].body/type/documents/snippets/exact_answer`); registration required at participants-area.bioasq.org | `dataset: {name: bioasq, path: ...}` | document-level only | Abstracts are not in the files: supply a PubMed-derived JSONL corpus (`document_id` = PMID) or use `corpus.source: bioasq_snippets` (snippets only; closed pool, optimistic). Test files may carry no answers/evidence -> left empty. |
| MIRAGE | `benchmark.json` | `dataset: {name: mirage, subset: <key>, path: ...}` | none | Multiple-choice; no corpus provided; most subsets are not multi-hop; no official train/dev split. |
| MedHopQA | JSON/JSONL from the official release | `dataset: {name: medhopqa, path: ..., field_map: {...}}` | if mapped | Schema unverified. Use its own source corpus as JSONL; never swap in unrelated PubMed. Do not compare its scores with PubMed benchmarks. |

## Corpus JSONL format
One object per line: `{"document_id": "...", "title": "...", "text": "...", "source": "...", "url": null, "metadata": {}}`.
Empty / duplicate-ID records are skipped and listed in the index manifest (`load_failures`).

## Leakage rules enforced in code
The pipeline reads only the question (and MCQ options). Gold answers/evidence are copied into `predictions.jsonl` solely for scoring.
Tune thresholds on a validation split only; `sweep` refuses `split: test` unless `--allow-test` (demo data only).
