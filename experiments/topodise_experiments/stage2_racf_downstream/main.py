"""Formal Stage2 entry; implementation lives in the shared trainer."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from topodise.training.configs import Stage2Config

def run_single_experiment(cfg):
    from topodise.training.stage2 import run_single_experiment as run
    return run(cfg)

def run_paper_experiment(exp, output_dir, device='cpu', num_workers=4):
    cfg = Stage2Config(**exp.get('stage2_overrides', {}))
    cfg.output_dir, cfg.device, cfg.num_workers = output_dir, device, num_workers
    return run_single_experiment(cfg)

if __name__ == '__main__':
    from experiments._execution import single_main
    single_main('experiments.topodise_experiments.stage2_racf_downstream.main')
