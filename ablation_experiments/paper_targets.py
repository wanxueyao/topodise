"""Full-system Stage0 × Stage1 target comparison in the paper."""
from pathlib import Path

from topodise.training.runtime.batch_experiments import main as run_stages
from topodise.training.runtime.full_stage2 import TASKS


ROUTES = {
    f"{stage0}_{stage1}": (stage0, stage1)
    for stage0 in ("gt", "hgrad_dual", "frangi")
    for stage1 in ("frangi", "hgrad_dual")
}


def main(argv=None):
    return run_stages(
        argv,
        routes=ROUTES,
        default_tasks=TASKS[:3],
        home=Path(__file__).resolve().parent,
        default_name="paper_pd_targets",
    )


if __name__ == "__main__":
    main()
