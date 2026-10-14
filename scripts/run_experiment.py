import sys
from adaptive_biorag.cli import main
sys.exit(main(["run", *sys.argv[1:]]))
