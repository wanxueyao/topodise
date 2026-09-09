"""basic 实验控制：编辑下方最佳配置、RUN_GROUPS 和 AVAILABLE_GPUS 后运行本文件。"""
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
OUTPUT_DIR = 'results/basic'
RUN_GROUPS = []  # 填写下方 _group 名称；[] 运行全部保留配置
MAIN_MODULE = 'experiments.basic_experiments.main'

# ==================== 最佳实验配置 ====================
# 每组字典直接传给 main.py；修改这里的参数会原样传给训练进程。
EXPERIMENTS = [{'architecture': 'dual_stream',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'epochs': 40,
  'learning_rate': 0.0002,
  'weight_decay': 0.003,
  'backbone_lr_ratio': 0.05,
  'lr_scheduler': 'cosine',
  'early_stopping': True,
  'patience': 6,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 6.0,
  'aug_brightness': 0.12,
  'aug_contrast': 0.12,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_group': 'P11_resnet18_patient_acc_seed42',
  '_desc': 'ResNet18 reference patient-acc recheck on pd+stage, seed=42',
  'seed': 42,
  '_method': 'basic'},
 {'architecture': 'dual_stream',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'epochs': 40,
  'learning_rate': 0.0002,
  'weight_decay': 0.003,
  'backbone_lr_ratio': 0.05,
  'lr_scheduler': 'cosine',
  'early_stopping': True,
  'patience': 6,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 6.0,
  'aug_brightness': 0.12,
  'aug_contrast': 0.12,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_group': 'P12_resnet18_patient_acc_seed202',
  '_desc': 'ResNet18 reference patient-acc recheck on pd+stage, seed=202',
  'seed': 202,
  '_method': 'basic'},
 {'architecture': 'dual_stream',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'epochs': 40,
  'learning_rate': 0.0002,
  'weight_decay': 0.003,
  'backbone_lr_ratio': 0.05,
  'lr_scheduler': 'cosine',
  'early_stopping': True,
  'patience': 6,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 6.0,
  'aug_brightness': 0.12,
  'aug_contrast': 0.12,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_group': 'P13_resnet18_patient_acc_seed3407',
  '_desc': 'ResNet18 reference patient-acc recheck on pd+stage, seed=3407',
  'seed': 3407,
  '_method': 'basic'},
 {'architecture': 'dual_stream',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'epochs': 40,
  'learning_rate': 0.0002,
  'weight_decay': 0.003,
  'backbone_lr_ratio': 0.05,
  'lr_scheduler': 'cosine',
  'early_stopping': True,
  'patience': 6,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 6.0,
  'aug_brightness': 0.12,
  'aug_contrast': 0.12,
  '_run_tasks': ['dm_binary'],
  '_group': 'P21_resnet18_dm_patient_acc_seed42',
  '_desc': 'ResNet18 reference patient-acc recheck on dm_binary, seed=42',
  'seed': 42,
  '_method': 'basic',
  '_transfer': 'dm',
  '_transfer_source': 'imagenet'},
 {'architecture': 'dual_stream',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'epochs': 40,
  'learning_rate': 0.0002,
  'weight_decay': 0.003,
  'backbone_lr_ratio': 0.05,
  'lr_scheduler': 'cosine',
  'early_stopping': True,
  'patience': 6,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 6.0,
  'aug_brightness': 0.12,
  'aug_contrast': 0.12,
  '_run_tasks': ['dm_binary'],
  '_group': 'P22_resnet18_dm_patient_acc_seed202',
  '_desc': 'ResNet18 reference patient-acc recheck on dm_binary, seed=202',
  'seed': 202,
  '_method': 'basic',
  '_transfer': 'dm',
  '_transfer_source': 'imagenet'},
 {'architecture': 'dual_stream',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'epochs': 40,
  'learning_rate': 0.0002,
  'weight_decay': 0.003,
  'backbone_lr_ratio': 0.05,
  'lr_scheduler': 'cosine',
  'early_stopping': True,
  'patience': 6,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 6.0,
  'aug_brightness': 0.12,
  'aug_contrast': 0.12,
  '_run_tasks': ['dm_binary'],
  '_group': 'P23_resnet18_dm_patient_acc_seed3407',
  '_desc': 'ResNet18 reference patient-acc recheck on dm_binary, seed=3407',
  'seed': 3407,
  '_method': 'basic',
  '_transfer': 'dm',
  '_transfer_source': 'imagenet'},
 {'architecture': 'dual_stream',
  'pretrained': 'Half',
  'epochs': 30,
  'learning_rate': 0.0002,
  'weight_decay': 0.003,
  'backbone_lr_ratio': 0.05,
  'lr_scheduler': 'cosine',
  'early_stopping': True,
  'patience': 6,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 6.0,
  'aug_brightness': 0.12,
  'aug_contrast': 0.12,
  'num_folds': 5,
  'kfold_repeats': 3,
  'seed': 42,
  '_group': 'C02_subtype_td_rest_resnet18',
  '_desc': 'PD subtype: TD=0 vs Rest (PIGD+mixed)=1; retained TD-rest branch parameters and patient '
           'folds',
  'backbone': 'resnet18',
  '_method': 'basic',
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
    run_batch(MAIN_MODULE, experiments, gpus=args.gpus,
              concurrency_per_gpu=args.concurrency_per_gpu,
              output_dir=args.output_dir, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
