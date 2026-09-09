"""
多任务学习数据加载
=================

支持:
1. CORN-1 数据集 (神经骨架分割)
2. PD 数据集 (PD分类)

特点:
- K-Fold 交叉验证 (患者级别)
- 数据增强 (图像/掩模同步增强)
- 多任务抽样
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from typing import List, Dict, Tuple, Literal, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ==================== CORN-1 数据集 ====================

class CORNDataset(Dataset):
    """
    CORN-1 神经骨架分割数据集（修正版）

    实际目录结构::

        CORN-1/
        ├── images/
        │   ├── train_orig/   ← .tif 原图
        │   ├── val_orig/
        │   └── test_orig/
        └── labels/
            ├── train_label/  ← .tif 二值掩模
            ├── val_label/
            └── test_label/

    图像和掩模的几何增强（翻转/旋转）严格同步，仅对图像做色彩抖动。
    """

    # 子目录名映射
    _IMAGE_DIRS = {
        'train': 'train_orig',
        'val':   'val_orig',
        'test':  'test_orig',
    }
    _LABEL_DIRS = {
        'train': 'train_label',
        'val':   'val_label',
        'test':  'test_label',
    }

    def __init__(self, root_dir: str, split: str = 'train',
                 augment: bool = False, image_size: int = 256,
                 generate_pseudo: bool = False):
        """
        Parameters
        ----------
        root_dir : str
            CORN-1 数据集根目录（包含 images/ 和 labels/ 子目录）
        split : str
            子集选择: 'train', 'val', 'test'
        augment : bool
            是否启用几何+色彩增强（仅 train split 建议开启）
        image_size : int
            目标图像尺寸（高 = 宽 = image_size）
        generate_pseudo : bool
            若为 True，则在 __getitem__ 中用 Frangi 伪标签替代 GT 掩模
        """
        if split not in self._IMAGE_DIRS:
            raise ValueError(f"split 必须是 'train'/'val'/'test'，got: {split}")

        self.image_dir = os.path.join(root_dir, 'images', self._IMAGE_DIRS[split])
        self.label_dir = os.path.join(root_dir, 'labels', self._LABEL_DIRS[split])
        self.augment = augment
        self.image_size = image_size
        self.generate_pseudo = generate_pseudo

        # 预实例化 Frangi 滤波器，避免每次 __getitem__ 重建
        self._frangi = None
        if self.generate_pseudo:
            from .multitask_utils import FrangiFilter
            self._frangi = FrangiFilter(sigma_range=(1.0, 5.0), num_scales=20)

        # 支持 .tif / .png / .jpg / .bmp
        exts = ('.tif', '.tiff', '.png', '.jpg', '.bmp')
        self.filenames = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(exts)
        )
        if not self.filenames:
            raise RuntimeError(f"CORN-1 {split} split 中未找到图像: {self.image_dir}")

        logger.info(f"CORN-1 [{split}] {len(self.filenames)} 张图像，image_dir={self.image_dir}")

    # ------------------------------------------------------------------
    # 同步增强辅助方法
    # ------------------------------------------------------------------
    def _sync_augment(self, image: Image.Image, mask: Image.Image):
        """对图像和掩模应用相同的几何变换，仅对图像应用色彩抖动。"""
        # 水平翻转
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # 随机旋转 ±6°
        angle = random.uniform(-6.0, 6.0)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # 仅对图像做色彩抖动（掩模不做）
        image = transforms.ColorJitter(brightness=0.12, contrast=0.12)(image)

        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.filenames[idx]

        # 加载图像（RGB）
        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')

        # 加载掩模（L 灰度 → 二值化）
        # 掩模文件名与图像文件名相同（同为 .tif）
        label_path = os.path.join(self.label_dir, fname)
        if not os.path.exists(label_path):
            # 尝试 .png 后缀
            label_path = os.path.join(
                self.label_dir,
                os.path.splitext(fname)[0] + '.png'
            )
        mask = Image.open(label_path).convert('L')

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask  = mask.resize( (self.image_size, self.image_size), Image.NEAREST)

        # 数据增强（同步）
        if self.augment:
            image, mask = self._sync_augment(image, mask)

        # 转张量 + 归一化图像
        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t,
                               mean=[0.485, 0.456, 0.406],
                               std =[0.229, 0.224, 0.225])

        # 掩模二值化 [1, H, W]
        mask_t = TF.to_tensor(mask)          # [1, H, W]，值 0.0~1.0
        mask_t = (mask_t > 0.5).float()

        # 如果需要 Frangi 伪标签，替换 GT 掩模
        if self.generate_pseudo:
            img_np = np.array(image)          # (H, W, 3)
            pseudo = self._frangi(img_np)     # (H, W)
            pseudo = (pseudo > pseudo.mean()).astype(np.float32)
            mask_t = torch.from_numpy(pseudo).unsqueeze(0)  # [1, H, W]

        return {
            'image': image_t,
            'mask':  mask_t,
            'name':  fname,
        }


def build_corn1_loaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    augment_train: bool = True,
    image_size: int = 256,
    generate_pseudo: bool = False,
    val_generate_pseudo: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    使用 CORN-1 预定义的 train/val 划分构建数据加载器。

    Parameters
    ----------
    root_dir : str
        CORN-1 根目录
    batch_size : int
        批大小
    num_workers : int
        DataLoader 工作进程数
    augment_train : bool
        是否对训练集启用增强
    image_size : int
        目标尺寸
    generate_pseudo : bool
        训练集是否使用 Frangi 伪标签替代 GT 掩模
    val_generate_pseudo : bool, optional
        验证集的伪标签开关。默认（None）= generate_pseudo，即与训练目标保持一致。
        训练伪标签、验证用 GT 会造成目标分布不匹配，Dice 接近 0。
        若确需同时拿 GT Dice，使用 evaluate_against_gt 另外实现。

    Returns
    -------
    (train_loader, val_loader)
    """
    if val_generate_pseudo is None:
        val_generate_pseudo = generate_pseudo

    train_set = CORNDataset(root_dir, split='train',
                            augment=augment_train, image_size=image_size,
                            generate_pseudo=generate_pseudo)
    val_set   = CORNDataset(root_dir, split='val',
                            augment=False, image_size=image_size,
                            generate_pseudo=val_generate_pseudo)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        f"CORN-1 数据加载器: 训练={len(train_set)}, 验证={len(val_set)}"
        f"  batch_size={batch_size}, pseudo={generate_pseudo}"
    )
    return train_loader, val_loader


# ==================== PD 数据集 ====================

class PDDataset(Dataset):
    """
    PD 分类数据集
    
    同时支持多任务学习 (可选的分割掩模)
    """
    
    def __init__(self, patients: List[Dict], images_dir: str,
                 include_masks: bool = False, masks_dir: str = None,
                 arch_mode: Literal['single_stream', 'dual_stream', 'mixed_single_stream'] = 'dual_stream',
                 transform: Optional[transforms.Compose] = None,
                 augment: bool = False, image_size: int = 256):
        """
        Parameters
        ----------

        patients : List[Dict]
            患者信息列表
        images_dir : str
            图像目录
        include_masks : bool
            是否包含分割掩模
        masks_dir : str, optional
            掩模目录
        arch_mode : str
            架构模式
        transform : Optional[transforms.Compose]
            图像变换
        augment : bool
            是否启用数据增强
        image_size : int
            目标图像尺寸
        """
        self.patients = patients
        self.images_dir = images_dir
        self.include_masks = include_masks
        self.masks_dir = masks_dir
        self.arch_mode = arch_mode
        
        # 基础变换
        base_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        
        # 数据增强
        if augment:
            base_transform = [
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(p=0.5),
            ] + base_transform
        
        # 归一化
        base_transform.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        )
        
        self.transform = transforms.Compose(base_transform) if transform is None else transform
    
    def __len__(self) -> int:
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns
        -------
        Dict
            包含 'image', 'pd_label', 'stage_label', 'subtype_label'
            以及可选的 'mask' 的字典
        """
        patient = self.patients[idx]
        
        # 加载图像
        image_path = os.path.join(self.images_dir, patient['image_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 标签编码
        pd_label = 1 if patient['group'] == 'pd' else 0
        stage_label = patient.get('stage', -1) - 1 if pd_label == 1 else -1
        
        # 亚型编码
        subtype_map = {'TD': 0, 'PIGD': 1, 'mixed': 2}
        subtype_label = subtype_map.get(patient.get('subtype', ''), -1) if pd_label == 1 else -1
        
        item = {
            'image': image,
            'pd_label': torch.tensor(pd_label, dtype=torch.long),
            'stage_label': torch.tensor(stage_label, dtype=torch.long),
            'subtype_label': torch.tensor(subtype_label, dtype=torch.long),
            'patient_id': patient.get('patient_id', ''),
        }
        
        # 可选的掩模
        if self.include_masks and self.masks_dir:
            mask_name = os.path.splitext(patient['image_path'])[0] + '.png'
            mask_path = os.path.join(self.masks_dir, mask_name)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = transforms.Resize((256, 256))(mask)
                mask = transforms.ToTensor()(mask)
                mask = (mask > 0.5).float()
                item['mask'] = mask
        
        return item


# ==================== 数据加载器构建 ====================



# ==================== PD 数据集（MTL 专用）====================

_DATASET_GROUPS = {
    'pd_control': ['pd', 'control'],
    'pd_control_cataract': ['pd', 'control_cataract'],
    'pd_all': ['pd', 'control', 'control_cataract'],
}

_SUBTYPE_MAP = {'TD': 0, 'PIGD': 1, 'mixed': 2}


class PDImageDataset(Dataset):
    """
    PD 图像数据集（多任务学习专用，每条样本为单张图像）。

    从 patients_dataset_en.json 加载患者列表，将每位患者的 central 和
    whorl 图像展开为独立样本。

    每个样本返回：
        image        : torch.Tensor [3, H, W]  归一化图像
        pd_label     : int  (0=控制组, 1=PD)
        stage_label  : int  (0‒4 HY 分期，控制组为 -1)
        subtype_label: int  (0=TD, 1=PIGD, 2=mixed，控制组为 -1)
        patient_idx  : int  用于患者级别 kfold

    Parameters
    ----------
    patients : List[Dict]
        过滤后的患者列表
    datasets_base : str
        图像根目录（json 中路径的前缀），通常为 'datasets'
    augment : bool
        是否启用数据增强
    image_size : int
        目标图像尺寸
    """

    def __init__(self, patients: List[Dict], datasets_base: str = 'datasets',
                 augment: bool = False, image_size: int = 256):
        self.datasets_base = datasets_base
        self.augment = augment
        self.image_size = image_size
        self.patients = patients

        # 展开为 (image_path, pd_label, stage_label, subtype_label, patient_idx)
        self.samples: List[Tuple] = []

        for p_idx, patient in enumerate(patients):
            is_pd = patient['group'] == 'pd'
            pd_l = 1 if is_pd else 0
            stage_l = (patient.get('stage', 0) - 1
                       if is_pd and patient.get('stage', 0) > 0 else -1)
            subtype_l = (_SUBTYPE_MAP.get(patient.get('subtype', ''), -1)
                         if is_pd else -1)

            paths = patient.get('image_paths', {})
            for region in ('central', 'whorl'):
                for rel_path in paths.get(region, []):
                    full_path = os.path.join(datasets_base, rel_path)
                    if os.path.exists(full_path):
                        self.samples.append(
                            (full_path, pd_l, stage_l, subtype_l, p_idx)
                        )

        if not self.samples:
            raise RuntimeError(
                f"PDImageDataset: 未找到任何有效图像。请确认 datasets_base={datasets_base!r}"
            )

        # 构建增强变换
        aug_ops = []
        if augment:
            aug_ops += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(6.0),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
            ]
        aug_ops += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(aug_ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, pd_l, stage_l, subtype_l, p_idx = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image_t = self.transform(image)
        return {
            'image':         image_t,
            'pd_label':      torch.tensor(pd_l,      dtype=torch.long),
            'stage_label':   torch.tensor(stage_l,   dtype=torch.long),
            'subtype_label': torch.tensor(subtype_l, dtype=torch.long),
            'patient_idx':   p_idx,
        }


def _patient_kfold_split(patients: List[Dict], fold_idx: int, n_folds: int,
                          random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    按患者做 K-fold 划分，保证同一患者的所有图像不跨 fold。

    先按 (group, stage) 分层，再均匀分配到各 fold。

    Returns
    -------
    train_patients, val_patients
    """
    rng = np.random.default_rng(random_state)

    # 按标签层分组
    strata: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(patients):
        is_pd = p['group'] == 'pd'
        stage = p.get('stage', 0) if is_pd else 0
        strata[f"pd{int(is_pd)}_s{stage}"].append(i)

    val_set: set = set()
    for indices in strata.values():
        idx_arr = np.array(indices)
        rng.shuffle(idx_arr)
        fold_size = max(1, len(idx_arr) // n_folds)
        v_start = fold_idx * fold_size
        v_end   = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else len(idx_arr)
        for i in idx_arr[v_start:v_end]:
            val_set.add(i)

    train_patients = [p for i, p in enumerate(patients) if i not in val_set]
    val_patients   = [p for i, p in enumerate(patients) if i in val_set]
    return train_patients, val_patients


def build_pd_patient_kfold_loaders(
    dataset_json: str,
    fold_idx: int,
    n_folds: int = 5,
    datasets_base: str = 'datasets',
    dataset_mode: str = 'pd_all',
    batch_size: int = 16,
    num_workers: int = 4,
    augment_train: bool = True,
    image_size: int = 256,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    从 patients_dataset_en.json 按患者做 K-fold 划分，返回 (train_loader, val_loader)。

    Parameters
    ----------
    dataset_json : str
        patients_dataset_en.json 路径（相对或绝对均可）
    fold_idx : int
        当前 fold 编号（0 ‒ n_folds-1）
    n_folds : int
        总 fold 数
    datasets_base : str
        图像文件路径前缀目录（'datasets'）
    dataset_mode : str
        过滤模式：'pd_control' / 'pd_control_cataract' / 'pd_all'
    batch_size : int
        批大小
    num_workers : int
        DataLoader 工作进程数
    augment_train : bool
        是否对训练集启用增强
    image_size : int
        目标图像尺寸
    random_state : int
        随机种子（确保各 fold 可复现）

    Returns
    -------
    (train_loader, val_loader)
    """
    with open(dataset_json, 'r', encoding='utf-8') as f:
        all_patients = json.load(f)

    groups = _DATASET_GROUPS.get(dataset_mode, _DATASET_GROUPS['pd_all'])
    patients = [p for p in all_patients if p['group'] in groups]

    train_patients, val_patients = _patient_kfold_split(
        patients, fold_idx, n_folds, random_state
    )

    train_set = PDImageDataset(
        train_patients, datasets_base=datasets_base,
        augment=augment_train, image_size=image_size,
    )
    val_set = PDImageDataset(
        val_patients, datasets_base=datasets_base,
        augment=False, image_size=image_size,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        f"PD 患者 kfold [{fold_idx}/{n_folds}]: "
        f"训练患者={len(train_patients)}→样本={len(train_set)}, "
        f"验证患者={len(val_patients)}→样本={len(val_set)}"
    )
    return train_loader, val_loader




class MultiTaskDataLoader:
    """
    多任务数据加载器（交替迭代 CORN-1 与 PD 批次）。

    每次迭代时同时产出一个 CORN-1 批次（task='segmentation'）和
    一个 PD 批次（task='classification'），调用方可按需使用。
    当较短的数据集耗尽后自动重置。
    """

    def __init__(self, corn1_loader: DataLoader, pd_loader: DataLoader):
        self.corn1_loader = corn1_loader
        self.pd_loader    = pd_loader

    def __len__(self) -> int:
        """以较长数据集的 batch 数为准"""
        return max(len(self.corn1_loader), len(self.pd_loader))

    def __iter__(self):
        """每次 yield 一个 (corn1_batch, pd_batch) 元组"""
        corn1_iter = iter(self.corn1_loader)
        pd_iter    = iter(self.pd_loader)

        for _ in range(len(self)):
            # CORN-1 批次
            try:
                corn1_batch = next(corn1_iter)
            except StopIteration:
                corn1_iter  = iter(self.corn1_loader)
                corn1_batch = next(corn1_iter)
            corn1_batch['task'] = 'segmentation'

            # PD 批次
            try:
                pd_batch = next(pd_iter)
            except StopIteration:
                pd_iter  = iter(self.pd_loader)
                pd_batch = next(pd_iter)
            pd_batch['task'] = 'classification'

            yield corn1_batch, pd_batch
