"""Thin wrapper (validation report; exit code 1 if errors): python data/validate_dataset.py --config configs/default.yaml"""
import sys
from adaptive_biorag.cli import main
sys.exit(main(["inspect-dataset", *sys.argv[1:]]))
