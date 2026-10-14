"""Thin wrapper: python data/build_corpus.py --config configs/default.yaml  (chunk + dedup + manifest + embeddings cache)"""
import sys
from adaptive_biorag.cli import main
sys.exit(main(["build-index", *sys.argv[1:]]))
