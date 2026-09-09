"""prototype 实验控制：编辑下方最佳配置、RUN_GROUPS 和 AVAILABLE_GPUS 后运行本文件。"""
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
OUTPUT_DIR = 'results/prototype'
RUN_GROUPS = []  # 填写下方 _group 名称；[] 运行全部保留配置
MAIN_MODULE = 'experiments.prototype_experiments.main'

# ==================== 最佳实验配置 ====================
# 每组字典直接传给 main.py；修改这里的参数会原样传给训练进程。
EXPERIMENTS = [{'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'embedding_dim': 128,
  'feature_dropout': 0.7,
  'n_way': 2,
  'n_shot': 3,
  'n_query': 4,
  'train_episodes_per_epoch': 80,
  'num_epochs': 24,
  'learning_rate': 5e-06,
  'weight_decay': 0.03,
  'lr_scheduler': 'cosine',
  'kfold_mode': 'patient',
  '_group': 'P11_e503_pd_patient_acc_seed42',
  '_desc': 'Prototype patient-acc recheck: 0625 E503/G700 pd line, seed=42',
  'patience': 2,
  'num_folds': 5,
  '_run_tasks': ['pd_binary'],
  'val_episodes': 200,
  'seed': 42,
  '_method': 'prototype'},
 {'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'embedding_dim': 128,
  'feature_dropout': 0.7,
  'n_way': 2,
  'n_shot': 3,
  'n_query': 4,
  'train_episodes_per_epoch': 80,
  'num_epochs': 24,
  'learning_rate': 5e-06,
  'weight_decay': 0.03,
  'lr_scheduler': 'cosine',
  'kfold_mode': 'patient',
  '_group': 'P12_e503_pd_patient_acc_seed202',
  '_desc': 'Prototype patient-acc recheck: 0625 E503/G700 pd line, seed=202',
  'patience': 2,
  'num_folds': 5,
  '_run_tasks': ['pd_binary'],
  'val_episodes': 200,
  'seed': 202,
  '_method': 'prototype'},
 {'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'embedding_dim': 128,
  'feature_dropout': 0.7,
  'n_way': 2,
  'n_shot': 3,
  'n_query': 4,
  'train_episodes_per_epoch': 80,
  'num_epochs': 24,
  'learning_rate': 5e-06,
  'weight_decay': 0.03,
  'lr_scheduler': 'cosine',
  'kfold_mode': 'patient',
  '_group': 'P13_e503_pd_patient_acc_seed3407',
  '_desc': 'Prototype patient-acc recheck: 0625 E503/G700 pd line, seed=3407',
  'patience': 2,
  'num_folds': 5,
  '_run_tasks': ['pd_binary'],
  'val_episodes': 200,
  'seed': 3407,
  '_method': 'prototype'},
 {'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'embedding_dim': 128,
  'feature_dropout': 0.55,
  'n_way': 3,
  'n_shot': 5,
  'n_query': 4,
  'train_episodes_per_epoch': 80,
  'num_epochs': 35,
  'learning_rate': 5e-05,
  'weight_decay': 0.01,
  'lr_scheduler': 'cosine',
  'kfold_mode': 'patient',
  '_group': 'P21_h816_stage_patient_acc_seed42',
  '_desc': 'Prototype patient-acc recheck: 0627 H816 stage line, seed=42',
  'patience': 4,
  'num_folds': 5,
  '_run_tasks': ['pd_stage'],
  'val_episodes': 200,
  'seed': 42,
  '_method': 'prototype'},
 {'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'embedding_dim': 128,
  'feature_dropout': 0.55,
  'n_way': 3,
  'n_shot': 5,
  'n_query': 4,
  'train_episodes_per_epoch': 80,
  'num_epochs': 35,
  'learning_rate': 5e-05,
  'weight_decay': 0.01,
  'lr_scheduler': 'cosine',
  'kfold_mode': 'patient',
  '_group': 'P22_h816_stage_patient_acc_seed202',
  '_desc': 'Prototype patient-acc recheck: 0627 H816 stage line, seed=202',
  'patience': 4,
  'num_folds': 5,
  '_run_tasks': ['pd_stage'],
  'val_episodes': 200,
  'seed': 202,
  '_method': 'prototype'},
 {'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'embedding_dim': 128,
  'feature_dropout': 0.55,
  'n_way': 3,
  'n_shot': 5,
  'n_query': 4,
  'train_episodes_per_epoch': 80,
  'num_epochs': 35,
  'learning_rate': 5e-05,
  'weight_decay': 0.01,
  'lr_scheduler': 'cosine',
  'kfold_mode': 'patient',
  '_group': 'P23_h816_stage_patient_acc_seed3407',
  '_desc': 'Prototype patient-acc recheck: 0627 H816 stage line, seed=3407',
  'patience': 4,
  'num_folds': 5,
  '_run_tasks': ['pd_stage'],
  'val_episodes': 200,
  'seed': 3407,
  '_method': 'prototype'},
 {'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'embedding_dim': 128,
  'feature_dropout': 0.65,
  'n_way': 2,
  'n_shot': 3,
  'n_query': 4,
  'train_episodes_per_epoch': 80,
  'num_epochs': 30,
  'learning_rate': 1e-05,
  'weight_decay': 0.01,
  'lr_scheduler': 'cosine',
  'kfold_mode': 'patient',
  '_group': 'I900_subtype_td_rest_d400_anchor',
  '_desc': 'PD subtype: TD=0 vs Rest (PIGD+mixed)=1; retained TD-rest branch parameters and patient '
           'folds',
  'patience': 3,
  'num_folds': 5,
  'kfold_repeats': 3,
  '_run_tasks': ['pd_subtype'],
  'val_episodes': 200,
  'seed': 42,
  '_method': 'prototype',
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
    run_batch(MAIN_MODULE, experiments, gpus=args.gpus,
              concurrency_per_gpu=args.concurrency_per_gpu,
              output_dir=args.output_dir, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
