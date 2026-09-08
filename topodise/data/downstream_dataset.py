"""Downstream PD / DM dataset for RACF stage 2.

PD has three sub-tasks:
- pd_binary  : control{,_cataract} vs pd            (2 classes)
- pd_stage   : H&Y stage 1/2/3                       (3 classes, pd-only)
- pd_subtype : TD / PIGD / mixed                     (3 classes, pd-only)

Subtype task-reconstruction variants (to bypass the tiny mixed-class collapse):
- pd_subtype_td_pigd : TD vs PIGD, mixed DROPPED      (2 classes, pd-only)
- pd_subtype_tdrest  : TD vs (PIGD+mixed merged)      (2 classes, pd-only)
- pd_subtype_pigd_mixed : PIGD vs mixed, TD DROPPED   (2 classes, pd-only)

Stage task-reconstruction variants:
- pd_stage_ge2 : stage 1 vs stage 2/3                 (2 classes, pd-only)
- pd_stage_ge3 : stage 1/2 vs stage 3                 (2 classes, pd-only)

Returns paired (x_central, x_whorl, y) drawn from the same patient.
"""
import os
import json
import random
from typing import List, Dict, Optional, Callable, Literal, Tuple

from PIL import Image
from torch.utils.data import Dataset


PD_TASKS = (
    "pd_binary", "pd_stage", "pd_subtype",
    "pd_subtype_td_pigd", "pd_subtype_tdrest", "pd_subtype_pigd_mixed",
    "pd_stage_ge2", "pd_stage_ge3",
)
PDTask = Literal[
    "pd_binary", "pd_stage", "pd_subtype",
    "pd_subtype_td_pigd", "pd_subtype_tdrest", "pd_subtype_pigd_mixed",
    "pd_stage_ge2", "pd_stage_ge3",
]

_SUBTYPE_MAP = {"TD": 0, "PIGD": 1, "mixed": 2}


# ---------------------------------------------------------------------------
# Shared decoded-image cache.
#
# The downstream datasets are tiny (~1500 grayscale 384x384 images, ~220MB when
# fully decoded) but every k-fold / epoch previously re-read and re-decoded the
# same files from disk hundreds of times, which dominated CPU time and starved
# the GPU. We decode each unique path exactly once and keep the PIL image in a
# process-global cache; the (possibly random) transform still runs per call, so
# augmentation behaviour is unchanged.
# ---------------------------------------------------------------------------
_GRAY_IMAGE_CACHE: Dict[str, "Image.Image"] = {}


def _load_gray(path: str, use_cache: bool = True) -> "Image.Image":
    """Return a decoded grayscale PIL image, caching the decode by path."""
    if not use_cache:
        return Image.open(path).convert("L")
    img = _GRAY_IMAGE_CACHE.get(path)
    if img is None:
        img = Image.open(path).convert("L")
        img.load()  # force decode now and release the underlying file handle
        _GRAY_IMAGE_CACHE[path] = img
    return img


def _patient_label(p: Dict, task: PDTask) -> Optional[int]:
    group = p.get("group")
    if task == "pd_binary":
        if group in ("control", "control_cataract"):
            return 0
        if group == "pd":
            return 1
        return None
    if task == "pd_stage":
        if group != "pd":
            return None
        stage = p.get("stage", 0)
        if stage in (1, 2, 3):
            return stage - 1
        return None
    if task == "pd_stage_ge2":
        if group != "pd":
            return None
        stage = p.get("stage", 0)
        if stage in (1, 2, 3):
            return 1 if stage >= 2 else 0
        return None
    if task == "pd_stage_ge3":
        if group != "pd":
            return None
        stage = p.get("stage", 0)
        if stage in (1, 2, 3):
            return 1 if stage >= 3 else 0
        return None
    if task == "pd_subtype":
        if group != "pd":
            return None
        return _SUBTYPE_MAP.get(p.get("subtype"))
    if task == "pd_subtype_td_pigd":
        # Feasibility-focused binary: TD vs PIGD, drop the ambiguous mixed class.
        if group != "pd":
            return None
        sub = p.get("subtype")
        if sub == "TD":
            return 0
        if sub == "PIGD":
            return 1
        return None  # mixed (and any unknown) excluded
    if task == "pd_subtype_tdrest":
        # All-patient binary: TD vs non-TD (PIGD + mixed merged into one class).
        if group != "pd":
            return None
        sub = p.get("subtype")
        if sub == "TD":
            return 0
        if sub in ("PIGD", "mixed"):
            return 1
        return None
    if task == "pd_subtype_pigd_mixed":
        # Second stage of a hierarchical subtype classifier: PIGD vs mixed,
        # after a TD-vs-rest gate. TD is excluded here.
        if group != "pd":
            return None
        sub = p.get("subtype")
        if sub == "PIGD":
            return 0
        if sub == "mixed":
            return 1
        return None
    raise ValueError(task)


class DownstreamDataset(Dataset):
    """Patient-level pairing: each sample is a (central_i, whorl_i) image pair drawn
    from the SAME patient, with the patient-level label.

    Pairing strategy: for each patient with K central and K whorl images we yield
    min(K_c, K_w) pairs (zip by index). If lengths differ we truncate to the shorter.
    """

    # Decode every unique image once (process-global cache); per-call transforms
    # still run so augmentation is unchanged. ``_final_cache`` is None for
    # training datasets (random transform) and set to a dict by the eval clone
    # builder, where the deterministic transform makes per-index tensor caching
    # safe and exact.
    cache_decoded: bool = True
    _final_cache: Optional[dict] = None

    def __init__(
        self,
        json_path: str,
        base_path: str,
        task: PDTask,
        transform: Optional[Callable] = None,
        patient_filter: Optional[Callable[[Dict], bool]] = None,
        pair_mode: Literal["zip", "cartesian"] = "zip",
        max_pairs_per_patient: int = 0,
        seed: int = 0,
    ):
        if task not in PD_TASKS:
            raise ValueError(f"task must be one of {PD_TASKS}, got {task}")
        self.task = task
        self.base_path = base_path
        self.transform = transform
        rng = random.Random(seed)

        with open(json_path, "r", encoding="utf-8") as f:
            patients = json.load(f)

        samples: List[Tuple[str, str, int, str]] = []
        labels_seen: Dict[int, int] = {}
        for p in patients:
            if patient_filter is not None and not patient_filter(p):
                continue
            y = _patient_label(p, task)
            if y is None:
                continue
            centrals = p.get("image_paths", {}).get("central", [])
            whorls = p.get("image_paths", {}).get("whorl", [])
            if pair_mode == "cartesian":
                pairs = [(c, w) for c in centrals for w in whorls]
            else:
                pairs = list(zip(centrals, whorls))
            if max_pairs_per_patient and len(pairs) > max_pairs_per_patient:
                rng.shuffle(pairs)
                pairs = pairs[:max_pairs_per_patient]
            for c, w in pairs:
                cp = os.path.join(base_path, c)
                wp = os.path.join(base_path, w)
                if os.path.exists(cp) and os.path.exists(wp):
                    samples.append((cp, wp, y, p.get("patient_id", "?")))
                    labels_seen[y] = labels_seen.get(y, 0) + 1
        if not samples:
            raise RuntimeError(
                f"DownstreamDataset[{task}] received zero samples. "
                f"json={json_path} base={base_path}"
            )
        self.samples = samples
        self.label_counts = labels_seen

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

    def few_shot_subset(self, k: int, seed: int = 0) -> "DownstreamDataset":
        """Return a new dataset with up to k samples per class."""
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
        clone.samples = new_samples
        clone.label_counts = {c: min(k, len(by_class[c])) for c in by_class}
        return clone


# ---------------------------------------------------------------------------
# Multi-task variant: produces all three PD labels per sample for joint training.
# Stratification key is the binary label (always defined).
# ---------------------------------------------------------------------------

# samples tuple layout: (central_path, whorl_path, y_bin, patient_id, y_stg, y_sub)
# y_stg / y_sub are -1 when not defined (control patients, or PD with unknown
# stage/subtype). y_bin is always 0 or 1 for kept samples.


def _patient_labels_multitask(p: Dict):
    """Return (y_bin, y_stg, y_sub) for one patient; y_bin may be None."""
    y_bin = _patient_label(p, "pd_binary")
    y_stg = _patient_label(p, "pd_stage")
    y_sub = _patient_label(p, "pd_subtype")
    return y_bin, (y_stg if y_stg is not None else -1), (y_sub if y_sub is not None else -1)


class MultiTaskDownstreamDataset(Dataset):
    """Joint dataset for {pd_binary, pd_stage, pd_subtype}.

    Each sample carries all three labels at once (missing -> -1). Suitable for
    multi-task heads where the loss for stage/subtype is masked out on samples
    where the corresponding label is unavailable.

    The ``samples`` attribute is a list of
    ``(central_path, whorl_path, y_bin, patient_id, y_stg, y_sub)``,
    preserving the ``(_, _, label, patient_id)`` layout that
    ``_build_kfold_splits`` in Stage 2 already relies on (stratification key is
    ``y_bin``).
    """

    cache_decoded: bool = True
    _final_cache: Optional[dict] = None

    def __init__(
        self,
        json_path: str,
        base_path: str,
        transform: Optional[Callable] = None,
        patient_filter: Optional[Callable[[Dict], bool]] = None,
    ):
        self.base_path = base_path
        self.transform = transform
        self.task = "pd_multitask"

        with open(json_path, "r", encoding="utf-8") as f:
            patients = json.load(f)

        samples: List[Tuple[str, str, int, str, int, int]] = []
        bin_counts: Dict[int, int] = {0: 0, 1: 0}
        stg_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, -1: 0}
        sub_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, -1: 0}
        for p in patients:
            if patient_filter is not None and not patient_filter(p):
                continue
            y_bin, y_stg, y_sub = _patient_labels_multitask(p)
            if y_bin is None:
                continue
            centrals = p.get("image_paths", {}).get("central", [])
            whorls = p.get("image_paths", {}).get("whorl", [])
            for c, w in zip(centrals, whorls):
                cp = os.path.join(base_path, c)
                wp = os.path.join(base_path, w)
                if os.path.exists(cp) and os.path.exists(wp):
                    samples.append((cp, wp, y_bin, p.get("patient_id", "?"), y_stg, y_sub))
                    bin_counts[y_bin] = bin_counts.get(y_bin, 0) + 1
                    stg_counts[y_stg] = stg_counts.get(y_stg, 0) + 1
                    sub_counts[y_sub] = sub_counts.get(y_sub, 0) + 1
        if not samples:
            raise RuntimeError(
                f"MultiTaskDownstreamDataset received zero samples. "
                f"json={json_path} base={base_path}"
            )
        self.samples = samples
        self.label_counts = bin_counts
        self.label_counts_bin = bin_counts
        self.label_counts_stg = {k: v for k, v in stg_counts.items() if k >= 0}
        self.label_counts_sub = {k: v for k, v in sub_counts.items() if k >= 0}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache = getattr(self, "_final_cache", None)
        if cache is not None:
            cached = cache.get(idx)
            if cached is not None:
                return cached
        cp, wp, y_bin, pid, y_stg, y_sub = self.samples[idx]
        use_cache = getattr(self, "cache_decoded", True)
        c = _load_gray(cp, use_cache)
        w = _load_gray(wp, use_cache)
        if self.transform:
            c, w = self.transform(c), self.transform(w)
        out = {
            "x_central": c, "x_whorl": w,
            "y_bin": y_bin, "y_stg": y_stg, "y_sub": y_sub,
            "y": y_bin,  # keep 'y' for any code that still expects single label
            "patient_id": pid,
        }
        if cache is not None:
            cache[idx] = out
        return out

    def few_shot_subset(self, k: int, seed: int = 0) -> "MultiTaskDownstreamDataset":
        """Reduce to up to k samples per binary class (kept simple)."""
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
        clone.samples = new_samples
        clone.label_counts = {c: min(k, len(by_class[c])) for c in by_class}
        clone.label_counts_bin = clone.label_counts
        clone.label_counts_stg = {}
        clone.label_counts_sub = {}
        return clone
