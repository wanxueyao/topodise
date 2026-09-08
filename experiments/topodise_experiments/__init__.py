"""TopoDiSE experiments: 3-stage pipeline + smoke tests.

- stage0_corn1_pretrain: SSL pretraining on CORN-1
- stage1_rscp_finetune : region-specialized continual pretraining
- stage2_racf_downstream: residual-adapter classifier finetune (PD: 3 sub-tasks)

DM tasks (e.g. dm_stage) are reserved as TODO; current focus is PD.
"""
