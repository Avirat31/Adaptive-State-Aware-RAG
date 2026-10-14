"""Thin wrapper: python data/inspect_dataset.py --config configs/default.yaml"""
import sys
from adaptive_biorag.cli import main
sys.exit(main(["inspect-dataset", *sys.argv[1:]]))
