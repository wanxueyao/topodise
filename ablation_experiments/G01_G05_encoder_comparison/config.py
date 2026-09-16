"""Only encoder/pretraining settings; all decoder recipes come from TopoDiSE."""
from copy import deepcopy
from pathlib import Path
import yaml
DEFAULTS = {'dataset_root': 'datasets',
 'tasks': ['pd_binary', 'pd_stage', 'pd_subtype_tdrest', 'dm_binary'],
 'repeats': 3,
 'bootstrap_samples': 2000,
 'bootstrap_seed': 20260915,
 'encoders': {'simclr_resnet18': {'checkpoint': 'ablation_experiments/G01_G05_encoder_comparison/results/pretrained/simclr_resnet18.pt',
                                  'auto_pretrain_if_missing': True,
                                  'pretrain': {'initialization': False,
                                               'batch_size': 32,
                                               'lr': 0.001,
                                               'temperature': 0.5,
                                               'proj_hidden_dim': 256,
                                               'proj_out_dim': 128,
                                               'weight_decay': 0.0001,
                                               'num_workers': 4,
                                               'amp': True,
                                               'channels_last': True,
                                               'persistent_workers': True,
                                               'pin_memory': False,
                                               'fused_adamw': True,
                                               'drop_last': True,
                                               'seed': 20260831,
                                               'stages': [{'name': 'stage_a_corn1',
                                                           'source': 'corn1_raw',
                                                           'source_root': 'datasets/CORN-1',
                                                           'epochs': 200,
                                                           'lr': 0.001},
                                                          {'name': 'stage_b_regions',
                                                           'source': 'region_manifest',
                                                           'manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                                                           'dataset_root': 'datasets',
                                                           'epochs': 80,
                                                           'lr': 0.0002}]}},
              'dinov2_vits14': {'checkpoint': 'ablation_experiments/G01_G05_encoder_comparison/pretrained/dinov2_vits14.pth',
                                'model_name': 'vit_small_patch14_dinov2'}}}

def load_config(path=None):
    result = deepcopy(DEFAULTS)
    if path:
        values = yaml.safe_load(Path(path).read_text()) or {}
        def merge(a,b):
            for k,v in b.items():
                if isinstance(v,dict) and isinstance(a.get(k),dict): merge(a[k],v)
                else: a[k]=v
        merge(result,values)
    return result
