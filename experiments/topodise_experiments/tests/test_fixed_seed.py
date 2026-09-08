import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.topodise_experiments.stage0_corn1_pretrain.config import Stage0Config
from experiments.topodise_experiments.stage1_rscp_finetune.config import Stage1Config
from topodise.reproducibility import configure_reproducibility


def test_fixed_seed_strategy_resets_torch_rng():
    first_state = configure_reproducibility(True)
    first = torch.rand(8)
    second_state = configure_reproducibility(True)
    second = torch.rand(8)

    assert torch.equal(first, second)
    assert first_state == second_state
    assert first_state['seed'] == 42
    assert first_state['strategy'] == 'fixed_torch_manual_seed'


def test_disabled_reproducibility_does_not_change_torch_rng():
    torch.manual_seed(7)
    _ = torch.rand(8)
    expected_next = torch.rand(8)

    torch.manual_seed(7)
    _ = torch.rand(8)
    state = configure_reproducibility(False)
    observed_next = torch.rand(8)

    assert torch.equal(expected_next, observed_next)
    assert state['seed'] is None
    assert state['strategy'] == 'environment_default'


def test_stage_configs_default_to_reproducibility_enabled():
    assert Stage0Config().reproducibility is True
    assert Stage1Config().reproducibility is True
