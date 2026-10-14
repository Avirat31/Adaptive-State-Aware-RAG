# Offline pilot on the synthetic example data (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m adaptive_biorag.cli build-index --config configs/default.yaml
python -m adaptive_biorag.cli run-suite --config configs/default.yaml --systems B0 B1 B2 B3 B4 B5 B6
