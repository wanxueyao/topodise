"""TopoDiSE Stage0 实验控制：编辑配置、实验选择及显卡后运行本文件。"""
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
OUTPUT_DIR = 'results/topodise/stage0'
RUN_GROUPS = []  # 留空运行本文件的全部配置
MAIN_MODULE = 'experiments.topodise_experiments.stage0_corn1_pretrain.main'

# 最佳 c8bf6b548656 配方；所有训练参数在此直接修改。
EXPERIMENTS = [{'_group': 'c8bf6b548656_stage0',
  'config': {'corn1_root': 'datasets/CORN-1',
             'seed': 42,
             'image_size': 224,
             'num_workers': 4,
             'persistent_workers': False,
             'prefetch_factor': 2,
             'pin_memory': None,
             'non_blocking_transfers': False,
             'expected_samples': 1516,
             'encoder_base_channels': 64,
             'encoder_channel_mult': (1, 2, 4, 8),
             'encoder_blocks_per_stage': (2, 2, 2, 2),
             'z_dim': 512,
             'unet_model_channels': 64,
             'unet_channel_mult': (1, 2, 2, 4),
             'T': 1000,
             'schedule': 'cosine',
             'epochs': 200,
             'batch_size': 16,
             'lr': 0.0001,
             'weight_decay': 0.01,
             'grad_clip': 1.0,
             'ema_decay': 0.9999,
             'z_dropout': 0.1,
             'lambda_simple': 1.0,
             'lambda_topo': 0.1,
             'topo_target': 'hgrad',
             'frangi_sigmas': (1.0, 2.0, 3.0),
             'frangi_beta': 0.5,
             'frangi_c': 0.5,
             'use_view_contrastive': False,
             'lambda_contrastive': 0.5,
             'tau_contrastive': 0.2,
             'output_dir': 'results/topodise',
             'run_name': 'c8bf6b548656_stage0',
             'resume': True,
             'resume_ckpt': '',
             'strict_resume': True,
             'device': 'auto',
             # strict=严格近似复刻；portable=设备自适应从零训练。
             'reproducibility': 'portable',
             'amp': False,
             'channels_last': False,
             'fused_adamw': False,
             'smoke': False}}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', '--groups', default=','.join(RUN_GROUPS), help='逗号分隔的实验名；留空运行全部')
    parser.add_argument('--gpus', default=os.getenv('RUN_AVAILABLE_GPUS', ','.join(map(str, AVAILABLE_GPUS)) or 'cpu'), help='如 0,1 使用两张卡；cpu 使用 CPU')
    parser.add_argument('--concurrency-per-gpu', type=int, default=CONCURRENCY_PER_GPU)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    repro_group = parser.add_mutually_exclusive_group()
    repro_group.add_argument(
        '--reproducibility', nargs='?', const='strict',
        choices=['strict', 'portable'],
        help='复现模式：strict 或 portable；不带值时等价于 strict',
    )
    repro_group.add_argument(
        '--no-reproducibility', dest='reproducibility', action='store_const',
        const='portable', help='兼容别名：等价于 --reproducibility portable',
    )
    parser.set_defaults(reproducibility=None)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.concurrency_per_gpu < 1:
        parser.error('num-workers 必须非负，concurrency-per-gpu 必须为正数')
    effective_reproducibility = str(
        args.reproducibility
        or os.environ.get('TOPODISE_REPRODUCIBILITY')
        or EXPERIMENTS[0]['config']['reproducibility']
    ).strip().lower()
    if effective_reproducibility in ('strict', '1', 'true', 'on', 'yes') and args.num_workers != 4:
        parser.error('strict 模式的 c8 近似复刻固定 --num-workers 4')
    experiments = copy.deepcopy(EXPERIMENTS)
    if args.reproducibility is not None:
        for exp in experiments:
            exp['config']['reproducibility'] = args.reproducibility
        # An explicit CLI choice must override an inherited environment value
        # in the worker process as well as the serialized config.
        os.environ['TOPODISE_REPRODUCIBILITY'] = args.reproducibility
        if args.reproducibility == 'strict':
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
    run_batch(MAIN_MODULE, experiments, gpus=args.gpus,
              concurrency_per_gpu=args.concurrency_per_gpu,
              output_dir=args.output_dir, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
