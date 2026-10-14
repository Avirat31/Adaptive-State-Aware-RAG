"""Thin wrapper kept for the target layout; passage preparation is part of build-index."""
import sys
from adaptive_biorag.cli import main
sys.exit(main(["build-index", *sys.argv[1:]]))
