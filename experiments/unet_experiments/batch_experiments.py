"""unet 实验控制：编辑下方最佳配置、RUN_GROUPS 和 AVAILABLE_GPUS 后运行本文件。"""
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
OUTPUT_DIR = 'results/unet'
RUN_GROUPS = []  # 填写下方 _group 名称；[] 运行全部保留配置
MAIN_MODULE = 'experiments.unet_experiments.main'
VARIANT = 'all'  # unet_only / unet_dual / all

# ==================== 最佳实验配置 ====================
# 每组字典直接传给 main.py；修改这里的参数会原样传给训练进程。
EXPERIMENTS = [{'_group': 'P01_c05_patient_acc_seed42',
  '_desc': 'UNet-only patient-acc recheck: C05 resnet18, seed=42, pd+stage only',
  'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'epochs': 45,
  'learning_rate': 0.00012,
  'weight_decay': 0.006,
  'lr_scheduler': 'cosine',
  'backbone_lr_ratio': 0.05,
  'early_stopping': True,
  'patience': 8,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 9.0,
  'aug_brightness': 0.14,
  'aug_contrast': 0.14,
  'seed': 42,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_method': 'unet_only'},
 {'_group': 'P02_c05_patient_acc_seed202',
  '_desc': 'UNet-only patient-acc recheck: C05 resnet18, seed=202, pd+stage only',
  'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'epochs': 45,
  'learning_rate': 0.00012,
  'weight_decay': 0.006,
  'lr_scheduler': 'cosine',
  'backbone_lr_ratio': 0.05,
  'early_stopping': True,
  'patience': 8,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 9.0,
  'aug_brightness': 0.14,
  'aug_contrast': 0.14,
  'seed': 202,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_method': 'unet_only'},
 {'_group': 'P03_c05_patient_acc_seed3407',
  '_desc': 'UNet-only patient-acc recheck: C05 resnet18, seed=3407, pd+stage only',
  'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'epochs': 45,
  'learning_rate': 0.00012,
  'weight_decay': 0.006,
  'lr_scheduler': 'cosine',
  'backbone_lr_ratio': 0.05,
  'early_stopping': True,
  'patience': 8,
  'batch_size': 8,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 9.0,
  'aug_brightness': 0.14,
  'aug_contrast': 0.14,
  'seed': 3407,
  '_run_tasks': ['pd_binary', 'pd_stage'],
  '_method': 'unet_only'},
 {'_group': 'E01_c05_subtype_td_rest_seed42',
  '_desc': 'PD subtype: TD=0 vs Rest (PIGD+mixed)=1; retained TD-rest branch parameters and patient '
           'folds',
  'kfold_repeats': 3,
  'backbone': 'resnet18',
  'architecture': 'dual_stream',
  'pretrained': 'Half',
  'epochs': 45,
  'learning_rate': 0.00012,
  'weight_decay': 0.006,
  'lr_scheduler': 'cosine',
  'backbone_lr_ratio': 0.03,
  'early_stopping': True,
  'patience': 8,
  'batch_size': 8,
  'dropout': 0.45,
  'pd_loss_type': 'label_smooth',
  'stage_loss_type': 'label_smooth',
  'subtype_loss_type': 'label_smooth',
  'label_smooth_eps': 0.12,
  'dataset_mode': 'pd_all',
  'kfold_mode': 'patient',
  'num_folds': 5,
  'seed': 42,
  'data_augmentation': True,
  'aug_horizontal_flip': 0.5,
  'aug_rotation': 9.0,
  'aug_brightness': 0.14,
  'aug_contrast': 0.14,
  '_method': 'unet_only',
  '_run_tasks': ['pd_subtype'],
  '_subtype_protocol': 'td_rest'},
 {'_group': 'P01_b04_patient_acc_seed42',
  '_desc': 'Dual-stream patient-acc recheck: B04 shared heavy aug, seed=42, pd+stage only',
  'name': 'P01_b04_patient_acc_seed42__shared_heavyaug_ep50_lr2e4_wd5e3_5fold',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'shared_backbone': True,
  'fusion_dim': 256,
  'learning_rate': 0.0002,
  'backbone_lr_ratio': 0.05,
  'weight_decay': 0.005,
  'epochs': 50,
  'patience': 10,
  'dropout': 0.4,
  'classifier_dropout': 0.4,
  'label_smoothing': 0.12,
  'aug_horizontal_flip': 0.5,
  'aug_vertical_flip': 0.2,
  'aug_rotation': 12.0,
  'aug_brightness': 0.18,
  'aug_contrast': 0.18,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'seed': 42,
  '_method': 'unet_dual',
  '_run_tasks': ['pd_binary', 'pd_stage']},
 {'_group': 'P02_b04_patient_acc_seed202',
  '_desc': 'Dual-stream patient-acc recheck: B04 shared heavy aug, seed=202, pd+stage only',
  'name': 'P02_b04_patient_acc_seed202__shared_heavyaug_ep50_lr2e4_wd5e3_5fold',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'shared_backbone': True,
  'fusion_dim': 256,
  'learning_rate': 0.0002,
  'backbone_lr_ratio': 0.05,
  'weight_decay': 0.005,
  'epochs': 50,
  'patience': 10,
  'dropout': 0.4,
  'classifier_dropout': 0.4,
  'label_smoothing': 0.12,
  'aug_horizontal_flip': 0.5,
  'aug_vertical_flip': 0.2,
  'aug_rotation': 12.0,
  'aug_brightness': 0.18,
  'aug_contrast': 0.18,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'seed': 202,
  '_method': 'unet_dual',
  '_run_tasks': ['pd_binary', 'pd_stage']},
 {'_group': 'P03_b04_patient_acc_seed3407',
  '_desc': 'Dual-stream patient-acc recheck: B04 shared heavy aug, seed=3407, pd+stage only',
  'name': 'P03_b04_patient_acc_seed3407__shared_heavyaug_ep50_lr2e4_wd5e3_5fold',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'shared_backbone': True,
  'fusion_dim': 256,
  'learning_rate': 0.0002,
  'backbone_lr_ratio': 0.05,
  'weight_decay': 0.005,
  'epochs': 50,
  'patience': 10,
  'dropout': 0.4,
  'classifier_dropout': 0.4,
  'label_smoothing': 0.12,
  'aug_horizontal_flip': 0.5,
  'aug_vertical_flip': 0.2,
  'aug_rotation': 12.0,
  'aug_brightness': 0.18,
  'aug_contrast': 0.18,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'seed': 3407,
  '_method': 'unet_dual',
  '_run_tasks': ['pd_binary', 'pd_stage']},
 {'_group': 'D01_shared_fd384_subtype_td_rest',
  '_desc': 'PD subtype: TD=0 vs Rest (PIGD+mixed)=1; retained TD-rest branch parameters and patient '
           'folds',
  'name': 'D01_shared_fd384_subtype_td_rest__dual_stream_Half_ep50_lr2e4_wd5e3_5x3',
  'backbone': 'resnet18',
  'pretrained': 'Half',
  'shared_backbone': True,
  'fusion_dim': 384,
  'learning_rate': 0.0002,
  'backbone_lr_ratio': 0.05,
  'weight_decay': 0.005,
  'epochs': 50,
  'patience': 10,
  'dropout': 0.4,
  'classifier_dropout': 0.4,
  'label_smoothing': 0.08,
  'aug_horizontal_flip': 0.5,
  'aug_vertical_flip': 0.0,
  'aug_rotation': 8.0,
  'aug_brightness': 0.14,
  'aug_contrast': 0.14,
  'num_folds': 5,
  'kfold_mode': 'patient',
  'seed': 42,
  'kfold_repeats': 3,
  '_method': 'unet_dual',
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
    parser.add_argument('--variant', choices=['unet_only', 'unet_dual', 'all'], default=VARIANT)
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.concurrency_per_gpu < 1:
        parser.error('num-workers 必须非负，concurrency-per-gpu 必须为正数')
    experiments = copy.deepcopy(EXPERIMENTS)
    experiments = [e for e in experiments if args.variant == 'all' or e['_method'] == args.variant]
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
