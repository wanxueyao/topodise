"""Validate the segmentation data required by both U-Net and GNN baselines."""
import json
from pathlib import Path

PREPARATION = '''先使用 CORN-1 训练 nnU-Net，再生成 PD 分割数据。在 codes/ 下执行：
  python -m unet.prepare_dataset
  python -m unet.train --profile baseline
  python -m unet.build_pd_unet_dataset
详见 unet/README.md。也可使用按此流程生成的完整 pd_dataset_unet_en 数据集。'''


def require_segmented_pd(data_root, require_original=False):
    root = Path(data_root)
    families = ['pd_dataset_unet_en']
    if require_original:
        families.append('pd_dataset_en')
    for family in families:
        manifest = root / family / 'patients_dataset_en.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'缺少 {manifest}\n{PREPARATION}')
        patients = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(patients, list) or not patients:
            raise ValueError(f'数据清单为空或格式错误: {manifest}\n{PREPARATION}')
        paths = [path for patient in patients for region in ('central', 'whorl')
                 for path in patient.get('image_paths', {}).get(region, [])]
        if not paths:
            raise ValueError(f'数据清单没有图像: {manifest}\n{PREPARATION}')
        missing = [str(root / path) for path in paths if not (root / path).is_file()]
        if missing:
            raise FileNotFoundError(f'缺少 {len(missing)} 张图像，例如 {missing[0]}\n{PREPARATION}')
