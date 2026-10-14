#!/usr/bin/env bash
# Offline pilot on the synthetic example data: builds the index, runs B0-B6, prints the results table.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m adaptive_biorag.cli build-index --config configs/default.yaml
python -m adaptive_biorag.cli run-suite --config configs/default.yaml --systems B0 B1 B2 B3 B4 B5 B6
