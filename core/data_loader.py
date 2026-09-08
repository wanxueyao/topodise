"""
数据加载模块

提供:
- PDDataset: 帕金森病图像数据集
- load_patients: 加载患者数据
- stratified_kfold: 多标签分层 K-fold（支持按患者或按图像分割）
"""
import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Literal, Generator, Tuple
from torch.utils.data import Dataset
from collections import defaultdict


# ==================== 类型定义 ====================

ArchMode = Literal['single_stream', 'mixed_single_stream', 'dual_stream']
DatasetMode = Literal['pd_control', 'pd_control_cataract', 'pd_all']
Region = Literal['central', 'whorl', 'all']
KFoldMode = Literal['patient', 'image']  # K-fold 划分模式：按患者或按图像


# ==================== 标签编码 ====================

SUBTYPE_MAP = {'TD': 0, 'PIGD': 1, 'mixed': 2}

DATASET_GROUPS = {
    'pd_control': ['pd', 'control'],
    'pd_control_cataract': ['pd', 'control_cataract'],
    'pd_all': ['pd', 'control', 'control_cataract']
}


def encode_labels(patient: Dict) -> Tuple[int, int, int]:
    """
    编码患者标签
    
    Returns: (pd_label, stage_label, subtype_label)
        - pd_label: 0=无PD, 1=有PD
        - stage_label: 0/1/2 或 -1(无效)
        - subtype_label: 0/1/2 或 -1(无效)
    """
    is_pd = patient['group'] == 'pd'
    pd_label = 1 if is_pd else 0
    stage_label = patient['stage'] - 1 if is_pd and patient['stage'] > 0 else -1
    subtype_label = SUBTYPE_MAP.get(patient['subtype'], -1) if is_pd else -1
    return pd_label, stage_label, subtype_label


# ==================== 数据集 ====================

class PDDataset(Dataset):
    """帕金森病图像数据集"""
    
    def __init__(self, patients: List[Dict], mode: ArchMode, transform=None,
                 region: Region = 'all', base_path: str = 'datasets'):
        """
        Args:
            patients: 患者数据列表
            mode: 架构模式 ('single_stream', 'mixed_single_stream', 'dual_stream')
            transform: 图像变换
            region: 区域过滤 (仅 single_stream 使用)
            base_path: 图像路径基础目录
        """
        self.mode = mode
        self.transform = transform
        self.samples: List = []
        self.PD_labels: List[int] = []
        self.stage_labels: List[int] = []
        self.subtype_labels: List[int] = []
        self.patients: List[Dict] = patients
        self.patient_ids: List[int] = []  # 追踪每个样本所属的患者 ID
        
        for patient_idx, patient in enumerate(patients):
            pd_l, stage_l, subtype_l = encode_labels(patient)
            paths = patient['image_paths']
            
            if mode == 'dual_stream':
                self._add_dual_samples(paths, pd_l, stage_l, subtype_l, base_path, patient_idx)
            else:
                images = self._get_images(paths, mode, region)
                self._add_single_samples(images, pd_l, stage_l, subtype_l, base_path, patient_idx)
    
    def _get_images(self, paths: Dict, mode: ArchMode, region: Region) -> List[str]:
        """获取图像路径列表"""
        if mode == 'single_stream':
            if region == 'central':
                return paths.get('central', [])
            elif region == 'whorl':
                return paths.get('whorl', [])
        return paths.get('central', []) + paths.get('whorl', [])
    
    def _add_single_samples(self, images: List[str], pd_l: int, stage_l: int, 
                            subtype_l: int, base_path: str, patient_idx: int):
        """添加单流样本"""
        for img in images:
            full_path = os.path.join(base_path, img)
            if os.path.exists(full_path):
                self.samples.append(full_path)
                self.PD_labels.append(pd_l)
                self.stage_labels.append(stage_l)
                self.subtype_labels.append(subtype_l)
                self.patient_ids.append(patient_idx)
    
    def _add_dual_samples(self, paths: Dict, pd_l: int, stage_l: int,
                          subtype_l: int, base_path: str, patient_idx: int):
        """添加双流样本"""
        central = paths.get('central', [])
        whorl = paths.get('whorl', [])
        for c, w in zip(central, whorl):
            c_path, w_path = os.path.join(base_path, c), os.path.join(base_path, w)
            if os.path.exists(c_path) and os.path.exists(w_path):
                self.samples.append({'central': c_path, 'whorl': w_path})
                self.PD_labels.append(pd_l)
                self.stage_labels.append(stage_l)
                self.subtype_labels.append(subtype_l)
                self.patient_ids.append(patient_idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        pd_l, stage_l, subtype_l = self.PD_labels[idx], self.stage_labels[idx], self.subtype_labels[idx]
        
        if self.mode == 'dual_stream':
            central = Image.open(self.samples[idx]['central']).convert('RGB')
            whorl = Image.open(self.samples[idx]['whorl']).convert('RGB')
            if self.transform:
                central, whorl = self.transform(central), self.transform(whorl)
            return (central, whorl), pd_l, stage_l, subtype_l
        else:
            image = Image.open(self.samples[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, pd_l, stage_l, subtype_l


# ==================== 数据加载 ====================

def load_and_filter_dataset(json_path: str, dataset_mode: DatasetMode) -> List[Dict]:
    """
    加载并过滤患者数据
    
    Args:
        json_path: JSON 数据文件路径
        dataset_mode: 数据集模式
    
    Returns:
        过滤后的患者列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        patients = json.load(f)
    
    groups = DATASET_GROUPS.get(dataset_mode)
    if not groups:
        raise ValueError(f"未知 dataset_mode: {dataset_mode}")
    
    return [p for p in patients if p['group'] in groups]


# ==================== K-Fold ====================

def stratified_kfold(pd_labels: List[int], stage_labels: List[int], 
                     subtype_labels: List[int], n_splits: int = 5, 
                     random_state: int = 42,
                     kfold_mode: KFoldMode = 'image',
                     patient_ids: List[int] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    多标签分层 K-fold
    
    确保每个 fold 中各标签组合的分布尽量均匀
    
    Args:
        pd_labels: PD 标签列表
        stage_labels: 分期标签列表  
        subtype_labels: 亚型标签列表
        n_splits: K-fold 的 K 值
        random_state: 随机种子
        kfold_mode: 划分模式
            - 'image': 按图像划分（默认，可能导致同一患者图像分布在不同fold）
            - 'patient': 按患者划分（避免数据泄露，同一患者的所有图像在同一fold）
        patient_ids: 每个样本对应的患者 ID 列表（仅 kfold_mode='patient' 时必须）
    
    Yields:
        (train_indices, val_indices)
    """
    if kfold_mode == 'patient':
        if patient_ids is None:
            raise ValueError("kfold_mode='patient' 需要提供 patient_ids 参数")
        yield from _patient_stratified_kfold(
            pd_labels, stage_labels, subtype_labels, patient_ids, n_splits, random_state
        )
    else:
        yield from _image_stratified_kfold(
            pd_labels, stage_labels, subtype_labels, n_splits, random_state
        )


def _image_stratified_kfold(pd_labels: List[int], stage_labels: List[int], 
                            subtype_labels: List[int], n_splits: int = 5, 
                            random_state: int = 42) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    按图像进行分层 K-fold（原有实现）
    
    警告: 同一患者的不同图像可能分布在训练集和验证集，存在数据泄露风险
    """
    rng = np.random.RandomState(random_state)
    
    # 按标签组合分组
    groups = defaultdict(list)
    for idx in range(len(pd_labels)):
        key = (pd_labels[idx], stage_labels[idx], subtype_labels[idx])
        groups[key].append(idx)
    
    # 初始化 fold
    folds = [[] for _ in range(n_splits)]
    
    # 轮询分配
    for indices in groups.values():
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(idx)
    
    # 打乱每个 fold
    for fold in folds:
        rng.shuffle(fold)
    
    # 生成划分
    for i in range(n_splits):
        val_idx = np.array(folds[i])
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, val_idx


def _patient_stratified_kfold(pd_labels: List[int], stage_labels: List[int], 
                              subtype_labels: List[int], patient_ids: List[int],
                              n_splits: int = 5, 
                              random_state: int = 42) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    按患者进行分层 K-fold
    
    确保同一患者的所有图像都在同一个 fold 中，避免数据泄露
    """
    rng = np.random.RandomState(random_state)
    
    # 构建患者信息：patient_id -> (样本索引列表, 标签)
    patient_info = defaultdict(lambda: {'indices': [], 'label': None})
    for idx in range(len(pd_labels)):
        pid = patient_ids[idx]
        patient_info[pid]['indices'].append(idx)
        # 所有来自同一患者的样本应有相同标签，取第一个即可
        if patient_info[pid]['label'] is None:
            patient_info[pid]['label'] = (pd_labels[idx], stage_labels[idx], subtype_labels[idx])
    
    # 按患者标签组合分组
    label_groups = defaultdict(list)  # label -> [patient_ids]
    for pid, info in patient_info.items():
        label_groups[info['label']].append(pid)
    
    # 初始化 fold（存储患者 ID）
    patient_folds = [[] for _ in range(n_splits)]
    
    # 轮询分配患者到各 fold
    for pids in label_groups.values():
        rng.shuffle(pids)
        for i, pid in enumerate(pids):
            patient_folds[i % n_splits].append(pid)
    
    # 打乱每个 fold 中的患者顺序
    for fold in patient_folds:
        rng.shuffle(fold)
    
    # 生成划分（将患者 ID 转换为样本索引）
    for i in range(n_splits):
        val_patients = patient_folds[i]
        train_patients = [pid for j in range(n_splits) if j != i for pid in patient_folds[j]]
        
        val_idx = np.array([idx for pid in val_patients for idx in patient_info[pid]['indices']])
        train_idx = np.array([idx for pid in train_patients for idx in patient_info[pid]['indices']])
        
        # 打乱样本顺序
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)
        
        yield train_idx, val_idx


# 兼容旧 API
_multilabel_stratified_kfold = stratified_kfold
