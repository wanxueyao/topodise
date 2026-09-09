"""gnn 实验控制：编辑下方最佳配置、RUN_GROUPS 和 AVAILABLE_GPUS 后运行本文件。"""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._execution import run_batch

# ==================== 在这里控制实验 ====================
AVAILABLE_GPUS = [0]  # [0] 一张卡；[0, 1] 两张卡；[] 使用 CPU
CONCURRENCY_PER_GPU = 1  # 每张卡同时运行几组实验；每组使用一张卡
NUM_WORKERS = 0
OUTPUT_DIR = 'results/gnn'
RUN_GROUPS = []  # 填写下方 _group 名称；[] 运行全部保留配置
MAIN_MODULE = 'experiments.gnn_experiments.main'

# ==================== 最佳实验配置 ====================
# 每组字典直接传给 main.py；修改这里的参数会原样传给训练进程。
EXPERIMENTS = [{'_group': 'P01_a01_patient_acc_seed42',
  '_desc': 'GNN patient-acc recheck: A01 GAT-64-L2, seed=42, patient 5-fold',
  'gnn_type': 'gat',
  'hidden_dim': 64,
  'num_layers': 2,
  'num_heads': 4,
  'shared_encoder': False,
  'lr': 0.001,
  'weight_decay': 0.0001,
  'optimizer': 'adam',
  'lr_scheduler': 'cosine',
  'warmup_epochs': 5,
  'epochs': 80,
  'patience': 15,
  'batch_size': 16,
  'dropout': 0.2,
  'classifier_dropout': 0.3,
  'pd_loss_weight': 1.0,
  'stage_loss_weight': 0.5,
  'subtype_loss_weight': 0.5,
  'label_smoothing': 0.0,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'dataset_group': 'pd_control',
  'seed': 42,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_method': 'gnn'},
 {'_group': 'P02_a01_patient_acc_seed202',
  '_desc': 'GNN patient-acc recheck: A01 GAT-64-L2, seed=202, patient 5-fold',
  'gnn_type': 'gat',
  'hidden_dim': 64,
  'num_layers': 2,
  'num_heads': 4,
  'shared_encoder': False,
  'lr': 0.001,
  'weight_decay': 0.0001,
  'optimizer': 'adam',
  'lr_scheduler': 'cosine',
  'warmup_epochs': 5,
  'epochs': 80,
  'patience': 15,
  'batch_size': 16,
  'dropout': 0.2,
  'classifier_dropout': 0.3,
  'pd_loss_weight': 1.0,
  'stage_loss_weight': 0.5,
  'subtype_loss_weight': 0.5,
  'label_smoothing': 0.0,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'dataset_group': 'pd_control',
  'seed': 202,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_method': 'gnn'},
 {'_group': 'P03_a01_patient_acc_seed3407',
  '_desc': 'GNN patient-acc recheck: A01 GAT-64-L2, seed=3407, patient 5-fold',
  'gnn_type': 'gat',
  'hidden_dim': 64,
  'num_layers': 2,
  'num_heads': 4,
  'shared_encoder': False,
  'lr': 0.001,
  'weight_decay': 0.0001,
  'optimizer': 'adam',
  'lr_scheduler': 'cosine',
  'warmup_epochs': 5,
  'epochs': 80,
  'patience': 15,
  'batch_size': 16,
  'dropout': 0.2,
  'classifier_dropout': 0.3,
  'pd_loss_weight': 1.0,
  'stage_loss_weight': 0.5,
  'subtype_loss_weight': 0.5,
  'label_smoothing': 0.0,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'dataset_group': 'pd_control',
  'seed': 3407,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_method': 'gnn'},
 {'_group': 'C01_gat64l2_subtype_td_rest',
  '_desc': 'PD subtype: TD=0 vs Rest (PIGD+mixed)=1; retained TD-rest branch parameters and patient '
           'folds',
  'gnn_type': 'gat',
  'hidden_dim': 64,
  'num_layers': 2,
  'num_heads': 4,
  'shared_encoder': False,
  'lr': 0.001,
  'weight_decay': 0.0001,
  'optimizer': 'adam',
  'lr_scheduler': 'cosine',
  'warmup_epochs': 5,
  'epochs': 80,
  'patience': 15,
  'batch_size': 16,
  'dropout': 0.2,
  'classifier_dropout': 0.3,
  'pd_loss_weight': 0.0,
  'stage_loss_weight': 0.0,
  'subtype_loss_weight': 1.0,
  'label_smoothing': 0.0,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'dataset_group': 'pd_control',
  'primary_task': 'subtype',
  'seed': 42,
  'kfold_repeats': 3,
  '_method': 'gnn',
  '_run_tasks': ['pd_subtype'],
  '_subtype_protocol': 'td_rest'}]


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
    from unet.prerequisites import require_segmented_pd
    require_segmented_pd(Path(__file__).resolve().parents[2] / 'datasets',
                         require_original=any(e['_method'] == 'unet_dual' for e in experiments))
    run_batch(MAIN_MODULE, experiments, gpus=args.gpus,
              concurrency_per_gpu=args.concurrency_per_gpu,
              output_dir=args.output_dir, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
