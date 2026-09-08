import os
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed):
    """设置随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def create_output_dir(base_dir, config):
    """
    创建输出目录
    
    格式：ARCHITECTURE_DATASET_BACKBONE_EPOCHS_YYYYMMDD_HHMMSS
    例如：single_stream_pd_control_resnet18_5_20260106_143000
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 简化架构名称
    arch_short = {
        'single_stream': 'single',
        'mixed_single_stream': 'mixed',
        'dual_stream': 'dual'
    }.get(config.ARCHITECTURE, config.ARCHITECTURE)
    
    # 简化数据集名称
    dataset_short = {
        'pd_control': 'pdc',
        'pd_control_cataract': 'pdcc',
        'pd_all': 'pdall'
    }.get(config.DATASET_MODE, config.DATASET_MODE)
    
    dir_name = f"{arch_short}_{dataset_short}_{config.BACKBONE}_{config.NUM_EPOCHS}_{timestamp}"
    output_dir = os.path.join(base_dir, dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(config, output_dir):
    """保存配置到文件"""
    config_file = os.path.join(output_dir, 'config.txt')
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("实验配置\n")
        f.write("=" * 50 + "\n\n")
        
        # 按字母顺序输出所有配置属性
        attrs = sorted([attr for attr in dir(config) if not attr.startswith('_') and attr.isupper()])
        for attr in attrs:
            value = getattr(config, attr)
            f.write(f"{attr}: {value}\n")
        
        f.write("=" * 50 + "\n")
