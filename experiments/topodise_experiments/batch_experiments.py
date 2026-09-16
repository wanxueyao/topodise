"""Main TopoDiSE: GT Stage0 + HGrad-dual Stage1 + four full downstream tasks."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.topodise_experiments.recipes import MAIN_TASKS_EXPERIMENTS, DM_TRANSFER_EXPERIMENTS
from topodise.training.runtime.batch_experiments import main
if __name__ == "__main__":
    main()
