"""DM (diabetic / cataract) downstream dataset.

DM JSON schema differs from PD: each patient has a single flat ``image_paths``
list (no central/whorl split). To feed our dual-branch encoder, we generate
(central, whorl) pairs from the same patient using one of two modes:

- ``duplicate``: same image fed to both branches.
- ``adjacent`` : pair image i with image i+1 from the same patient.

Tasks:
- ``dm_binary``: control (0) vs NDR/NPDR (1)
- ``dm_subtype``: NDR (0) vs NPDR (1); control samples are excluded
- ``dm_threeway``: control (0) vs NDR (1) vs NPDR (2); downstream reports can
    also evaluate subtype accuracy on the NDR/NPDR subset.
"""
import os
import json
import random
from typing import List, Optional, Callable, Literal, Tuple, Dict

from torch.utils.data import Dataset

from .downstream_dataset import _load_gray


DM_TASKS = ("dm_binary", "dm_subtype", "dm_threeway")
DMTask = Literal["dm_binary", "dm_subtype", "dm_threeway"]

_DM_SUBTYPE_MAP = {"NDR": 0, "NPDR": 1}
_DM_THREEWAY_MAP = {"control": 0, "NDR": 1, "NPDR": 2}


def _dm_label(group: str, task: DMTask) -> Optional[int]:
    if task == "dm_binary":
        if group == "control":
            return 0
        if group in ("NDR", "NPDR"):
            return 1
        return None
    if task == "dm_subtype":
        return _DM_SUBTYPE_MAP.get(group)
    if task == "dm_threeway":
        return _DM_THREEWAY_MAP.get(group)
    raise ValueError(task)


class DMDownstreamDataset(Dataset):
    # See DownstreamDataset: decode each image once (process-global cache); the
    # eval clone builder sets ``_final_cache`` to enable exact per-index tensor
    # caching under the deterministic eval transform.
    cache_decoded: bool = True
    _final_cache: Optional[dict] = None

    def __init__(
        self,
        json_path: str,
        base_path: str,
        task: DMTask,
        transform: Optional[Callable] = None,
        pair_mode: Literal["duplicate", "adjacent"] = "duplicate",
    ):
        if task not in DM_TASKS:
            raise ValueError(f"task must be one of {DM_TASKS}, got {task}")
        self.task = task
        self.base_path = base_path
        self.transform = transform
        self.pair_mode = pair_mode

        with open(json_path, "r", encoding="utf-8") as f:
            patients = json.load(f)

        samples: List[Tuple[str, str, int, str]] = []
        labels: Dict[int, int] = {}
        for p in patients:
            y = _dm_label(p.get("group", ""), task)
            if y is None:
                continue
            paths = p.get("image_paths", [])
            full_paths = [os.path.join(base_path, ip) for ip in paths]
            full_paths = [fp for fp in full_paths if os.path.exists(fp)]
            if not full_paths:
                continue
            pid = p.get("patient_id", "?")
            if pair_mode == "duplicate":
                for fp in full_paths:
                    samples.append((fp, fp, y, pid))
                    labels[y] = labels.get(y, 0) + 1
            elif pair_mode == "adjacent":
                if len(full_paths) < 2:
                    samples.append((full_paths[0], full_paths[0], y, pid))
                    labels[y] = labels.get(y, 0) + 1
                else:
                    for i in range(len(full_paths) - 1):
                        samples.append((full_paths[i], full_paths[i + 1], y, pid))
                        labels[y] = labels.get(y, 0) + 1
            else:
                raise ValueError(f"unknown pair_mode: {pair_mode}")

        if not samples:
            raise RuntimeError(
                f"DMDownstreamDataset[{task}] received zero samples. "
                f"json={json_path} base={base_path}"
            )
        self.samples = samples
        self.label_counts = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache = getattr(self, "_final_cache", None)
        if cache is not None:
            cached = cache.get(idx)
            if cached is not None:
                return cached
        cp, wp, y, pid = self.samples[idx]
        use_cache = getattr(self, "cache_decoded", True)
        c = _load_gray(cp, use_cache)
        w = _load_gray(wp, use_cache)
        if self.transform:
            c, w = self.transform(c), self.transform(w)
        out = {"x_central": c, "x_whorl": w, "y": y, "patient_id": pid}
        if cache is not None:
            cache[idx] = out
        return out

    def few_shot_subset(self, k: int, seed: int = 0) -> "DMDownstreamDataset":
        rng = random.Random(seed)
        by_class: Dict[int, list] = {}
        for s in self.samples:
            by_class.setdefault(s[2], []).append(s)
        new_samples = []
        for c, lst in by_class.items():
            rng.shuffle(lst)
            new_samples.extend(lst[:k])
        clone = self.__class__.__new__(self.__class__)
        clone.task = self.task
        clone.base_path = self.base_path
        clone.transform = self.transform
        clone.pair_mode = self.pair_mode
        clone.samples = new_samples
        clone.label_counts = {c: min(k, len(by_class[c])) for c in by_class}
        return clone
