"""
dual_view_dataset.py - 双视角图像数据集

将 pd_dataset_en（原始眼底图像）与 pd_dataset_unet_en（UNet 提取的神经纤维图像）
配对为同一患者同一解剖区域的两个视角，供双流 CNN 使用。

    每个样本：
    (en_img_tensor, unet_img_tensor, pd_label, stage_label, subtype_label, patient_id)
    en_img    来自 pd_dataset_en  (原始眼底)
    unet_img  来自 pd_dataset_unet_en (UNet 处理后)
    两张图对应同一患者的同一解剖区域（central 或 whorl）

数据匹配逻辑：
    对每个共同 patient_id，取 min(len(en_central), len(unet_central)) 个 central 对
    和 min(len(en_whorl), len(unet_whorl)) 个 whorl 对（受 region 参数控制）
"""

import os
from typing import List, Dict, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class DualViewDataset(Dataset):
    """
    双视角眼底图像数据集。

    Args:
        en_patients:   来自 pd_dataset_en 的患者列表（已过滤）
        unet_patients: 来自 pd_dataset_unet_en 的患者列表（与 en_patients 按 patient_id 对应）
        base_path:     图像路径基准目录（通常为 `codes/datasets`）
        en_transform:  应用于 en 图像的变换
        unet_transform: 应用于 unet 图像的变换（若为 None，复用 en_transform）
        region:        'both' | 'central' | 'whorl'
    """

    SUBTYPE_MAP = {'TD': 0, 'PIGD': 1, 'mixed': 2}

    def __init__(
        self,
        en_patients: List[Dict],
        unet_patients: List[Dict],
        base_path: str,
        en_transform=None,
        unet_transform=None,
        region: str = 'both',
    ):
        assert region in ('both', 'central', 'whorl'), \
            f"region 必须为 'both'/'central'/'whorl'，当前: {region}"
        assert len(en_patients) == len(unet_patients), \
            "en_patients 与 unet_patients 数量不匹配，请先按 patient_id 对齐"

        self.en_transform = en_transform
        self.unet_transform = unet_transform if unet_transform is not None else en_transform
        self.samples: List[Tuple[str, str, int, int, int, str]] = []  # (en_path, unet_path, pd, stage, sub, pid)

        for en_p, unet_p in zip(en_patients, unet_patients):
            pd_l, stage_l, subtype_l = self._encode(en_p)
            patient_id = str(en_p.get('patient_id', len(self.samples)))
            en_paths = en_p['image_paths']
            unet_paths = unet_p['image_paths']

            if region in ('both', 'central'):
                ec = en_paths.get('central', [])
                uc = unet_paths.get('central', [])
                for ep, up in zip(ec, uc):
                    ep_full = os.path.join(base_path, ep)
                    up_full = os.path.join(base_path, up)
                    if os.path.exists(ep_full) and os.path.exists(up_full):
                        self.samples.append((ep_full, up_full, pd_l, stage_l, subtype_l, patient_id))

            if region in ('both', 'whorl'):
                ew = en_paths.get('whorl', [])
                uw = unet_paths.get('whorl', [])
                for ep, up in zip(ew, uw):
                    ep_full = os.path.join(base_path, ep)
                    up_full = os.path.join(base_path, up)
                    if os.path.exists(ep_full) and os.path.exists(up_full):
                        self.samples.append((ep_full, up_full, pd_l, stage_l, subtype_l, patient_id))

    def _encode(self, patient: Dict) -> Tuple[int, int, int]:
        is_pd = patient['group'] == 'pd'
        pd_l = 1 if is_pd else 0
        stage_l = patient['stage'] - 1 if is_pd and patient.get('stage', 0) > 0 else -1
        subtype_l = self.SUBTYPE_MAP.get(patient.get('subtype') or '', -1) if is_pd else -1
        return pd_l, stage_l, subtype_l

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        en_path, unet_path, pd_l, stage_l, subtype_l, patient_id = self.samples[idx]
        en_img = Image.open(en_path).convert('RGB')
        unet_img = Image.open(unet_path).convert('RGB')
        if self.en_transform:
            en_img = self.en_transform(en_img)
        if self.unet_transform:
            unet_img = self.unet_transform(unet_img)
        return en_img, unet_img, pd_l, stage_l, subtype_l, patient_id
