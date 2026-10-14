import sys
from adaptive_biorag.cli import main
sys.exit(main(["evaluate", *sys.argv[1:]]))
