"""Shared experiment support and compatibility for the unchanged Stage2 project."""
import sys
from . import _stage_configs

# Preserve imports used by unified_stage2_encoder_comparison without restoring per-stage files.
for _stage in ("stage0_corn1_pretrain", "stage1_rscp_finetune", "stage2_racf_downstream"):
    sys.modules[f"{__name__}.topodise_experiments.{_stage}.config"] = _stage_configs
del _stage
