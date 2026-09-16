"""Run stage0 of the main GT/HGrad-dual pipeline."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from topodise.training.runtime.batch_experiments import main as run

def main(argv=None):
    return run(["--phase", "stage0"] + (sys.argv[1:] if argv is None else list(argv)))
if __name__ == "__main__":
    main()
