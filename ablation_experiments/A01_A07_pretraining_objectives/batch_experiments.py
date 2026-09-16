"""Numbered group entry; all options are shared with the top-level runner."""
import sys
from ablation_experiments.run import main as run

def main(argv=None):
    return run(["--experiments", "A", *(sys.argv[1:] if argv is None else argv)])

if __name__=="__main__": main()
