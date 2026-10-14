import sys
from adaptive_biorag.cli import main
sys.exit(main(["build-index", *sys.argv[1:]]))
