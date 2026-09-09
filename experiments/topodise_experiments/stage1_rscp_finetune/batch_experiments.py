"""TopoDiSE Stage1 实验控制：编辑配置、实验选择及显卡后运行本文件。"""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments._execution import run_batch

AVAILABLE_GPUS = [0]  # [0, 1] 使用两张卡；[] 使用 CPU；每组实验占一张卡
CONCURRENCY_PER_GPU = 1
NUM_WORKERS = 4
OUTPUT_DIR = 'results/topodise/stage1'
RUN_GROUPS = []  # 留空运行本文件的全部配置
MAIN_MODULE = 'experiments.topodise_experiments.stage1_rscp_finetune.main'

# 最佳 c8bf6b548656 配方；所有训练参数在此直接修改。
# config.parent_ckpt 必须填写 Stage0 实际输出的 ema_best.pt 路径。
EXPERIMENTS = [{'_group': 'c8bf6b548656_stage1',
  'config': {'parent_ckpt': '',
             'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
             'region_pair_manifest': None,
             'base_path': 'datasets',
             'patient_groups': ('control', 'control_cataract'),
             'allow_dm_disease': False,
             'image_size': 224,
             'num_workers': 4,
             'persistent_workers': False,
             'prefetch_factor': 2,
             'pin_memory': None,
             'non_blocking_transfers': False,
             'expected_region_pairs': 182,
             'epochs': 80,
             'batch_size': 16,
             'lr': 2e-05,
             'weight_decay': 0.01,
             'grad_clip': 1.0,
             'ema_decay': 0.9999,
             'lambda_simple': 1.0,
             'lambda_topo': 0.05,
             'lambda_region': 0.5,
             'tau_region': 0.2,
             'topo_target': 'hgrad',
             'frangi_sigmas': (1.0, 2.0, 3.0),
             'frangi_beta': 0.5,
             'frangi_c': 0.5,
             'train_denoiser': False,
             'output_dir': 'results/topodise',
             'run_name': 'c8bf6b548656_stage1',
             'device': 'auto',
             'reproducibility': True,
             'parent_encoder_weights': 'raw',
             'amp': False,
             'channels_last': False,
             'fused_adamw': False,
             'resume': True,
             'resume_ckpt': '',
             'strict_resume': True,
             'smoke': False}}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', '--groups', default=','.join(RUN_GROUPS), help='逗号分隔的实验名；留空运行全部')
    parser.add_argument('--gpus', default=os.getenv('RUN_AVAILABLE_GPUS', ','.join(map(str, AVAILABLE_GPUS)) or 'cpu'), help='如 0,1 使用两张卡；cpu 使用 CPU')
    parser.add_argument('--concurrency-per-gpu', type=int, default=CONCURRENCY_PER_GPU)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.concurrency_per_gpu < 1:
        parser.error('num-workers 必须非负，concurrency-per-gpu 必须为正数')
    experiments = copy.deepcopy(EXPERIMENTS)
    groups = [g.strip() for g in args.group.split(',') if g.strip()]
    unknown = set(groups) - {e['_group'] for e in experiments}
    if unknown:
        parser.error(f'Unknown groups: {sorted(unknown)}')
    experiments = [e for e in experiments if not groups or e['_group'] in groups]
    if not experiments:
        parser.error('No experiments selected')
    if args.list:
        for exp in experiments:
            print(exp['_group'], exp.get('_desc', ''))
        return
    if args.dry_run:
        print(json.dumps({'gpus': args.gpus, 'concurrency_per_gpu': args.concurrency_per_gpu,
                          'num_workers': args.num_workers, 'output_dir': args.output_dir,
                          'experiments': experiments}, indent=2, ensure_ascii=False))
        return
    from experiments._execution import output_path
    for exp in experiments:
        parent = output_path(exp['config']['parent_ckpt'])
        if not parent.is_file():
            parser.error(f'Stage1 需要先运行 Stage0，并把 config.parent_ckpt 改为其实际检查点路径：{parent}')
        exp['config']['parent_ckpt'] = str(parent)
    run_batch(MAIN_MODULE, experiments, gpus=args.gpus,
              concurrency_per_gpu=args.concurrency_per_gpu,
              output_dir=args.output_dir, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
