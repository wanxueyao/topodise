"""
multitask_utils.py 中缺少的辅助函数
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    设置随机种子确保可复现性
    
    Parameters
    ----------
    seed : int
        随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
