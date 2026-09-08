"""RSCP region-pair dataset: real (central, whorl) pairs from whitelisted roots.

Strict allow-list to avoid leaking PD-positive labels:
- unlabeled_dataset_en/
- pd_dataset_en/control/, pd_dataset_en/control_cataract/
- dm_dataset_en/control/

DM disease images (NDR/NPDR) are allowed only when explicitly requested for
DM-domain self-supervised adaptation; PD-positive images remain forbidden.
"""
import os
import json
from typing import List, Optional, Callable, Dict, Sequence, Set, Tuple
from PIL import Image
from torch.utils.data import Dataset


ALLOWED_PATH_FRAGMENTS = (
    "unlabeled_dataset_en",
    os.path.join("pd_dataset_en", "control"),
    os.path.join("pd_dataset_en", "control_cataract"),
    os.path.join("dm_dataset_en", "control"),
)

FORBIDDEN_FRAGMENTS = (
    os.path.join("pd_dataset_en", "pd"),
    os.path.join("dm_dataset_en", "NDR"),
    os.path.join("dm_dataset_en", "NPDR"),
)

DM_DISEASE_FRAGMENTS = (
    os.path.join("dm_dataset_en", "NDR"),
    os.path.join("dm_dataset_en", "NPDR"),
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
REFERENCE_DATASET_DIRS = ("pd_dataset_en", "dm_dataset_en")


def _is_allowed(path: str, allow_dm_disease: bool = False) -> bool:
    forbidden = tuple(
        frag for frag in FORBIDDEN_FRAGMENTS
        if allow_dm_disease is False or frag not in DM_DISEASE_FRAGMENTS
    )
    allowed = ALLOWED_PATH_FRAGMENTS
    if allow_dm_disease:
        allowed = allowed + DM_DISEASE_FRAGMENTS
    return any(frag in path for frag in allowed) and not any(frag in path for frag in forbidden)


def _is_unlabeled_path(path: str) -> bool:
    return "unlabeled_dataset_en" in os.path.normpath(path)


def _collect_reference_basenames(base_path: str,
                                 dataset_dirs: Tuple[str, ...] = REFERENCE_DATASET_DIRS) -> Set[str]:
    basenames: Set[str] = set()
    for dataset_dir in dataset_dirs:
        root = os.path.join(base_path, dataset_dir)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    basenames.add(filename.lower())
    return basenames


def _patient_to_pairs(p: Dict) -> List[Dict]:
    image_paths = p.get("image_paths", {})
    pid = p.get("patient_id", "?")
    out: List[Dict] = []
    if isinstance(image_paths, dict):
        centrals = image_paths.get("central", [])
        whorls = image_paths.get("whorl", [])
        if whorls:
            for c, w in zip(centrals, whorls):
                out.append({"subject_id": pid, "central": c, "whorl": w, "is_paired": True})
        else:
            for c in centrals:
                out.append({"subject_id": pid, "central": c, "whorl": c, "is_paired": False})
    elif isinstance(image_paths, Sequence) and not isinstance(image_paths, (str, bytes)):
        for c in image_paths:
            out.append({"subject_id": pid, "central": c, "whorl": c, "is_paired": False})
    return out


class RegionPairDataset(Dataset):
    """Reads paired (central, whorl) images from a manifest JSON.

    Manifest format (list of objects):
        {"subject_id": "...", "central": "<rel_path>", "whorl": "<rel_path>"}

    Or auto-build from a list of patient dicts with keys 'group' and 'image_paths'.
    Patient-style manifests are also accepted. If a patient has only central
    images (as in unlabeled_dataset_en), each central image is duplicated into
    both branches so Stage1 can still use the larger unlabeled pool without
    requiring region labels.
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        patients: Optional[List[Dict]] = None,
        base_path: str = "datasets",
        transform: Optional[Callable] = None,
        strict_whitelist: bool = True,
        allow_dm_disease: bool = False,
        exclude_unlabeled_duplicate_filenames: bool = True,
    ):
        self.base_path = base_path
        self.transform = transform
        self.strict_whitelist = strict_whitelist
        self.allow_dm_disease = allow_dm_disease
        self.exclude_unlabeled_duplicate_filenames = exclude_unlabeled_duplicate_filenames
        reference_basenames = (
            _collect_reference_basenames(base_path)
            if exclude_unlabeled_duplicate_filenames else set()
        )
        self.skipped_duplicate_filenames = 0
        pairs: List[Dict[str, str]] = []

        if manifest_path is not None:
            with open(manifest_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if rows and isinstance(rows[0], dict) and "image_paths" in rows[0]:
                for p in rows:
                    pairs.extend(_patient_to_pairs(p))
            else:
                pairs = rows
        elif patients is not None:
            for p in patients:
                pairs.extend(_patient_to_pairs(p))
        else:
            raise ValueError("must provide manifest_path or patients")

        # filter + validate against whitelist
        clean: List[Dict] = []
        for r in pairs:
            cp = os.path.join(base_path, r["central"])
            wp = os.path.join(base_path, r["whorl"])
            if not (os.path.exists(cp) and os.path.exists(wp)):
                continue
            if reference_basenames and (_is_unlabeled_path(cp) or _is_unlabeled_path(wp)):
                if (os.path.basename(cp).lower() in reference_basenames
                        or os.path.basename(wp).lower() in reference_basenames):
                    self.skipped_duplicate_filenames += 1
                    continue
            # Whitelist enforcement: ANY path that contains a forbidden fragment is dropped;
            # otherwise we require the allowed fragment to appear.
            full_c = os.path.abspath(cp)
            full_w = os.path.abspath(wp)
            if strict_whitelist:
                if not (_is_allowed(full_c, allow_dm_disease=allow_dm_disease)
                        and _is_allowed(full_w, allow_dm_disease=allow_dm_disease)):
                    continue
            else:
                # only enforce forbidden fragments (PD-positive leak)
                forbidden = tuple(
                    frag for frag in FORBIDDEN_FRAGMENTS
                    if allow_dm_disease is False or frag not in DM_DISEASE_FRAGMENTS
                )
                if any(frag in full_c or frag in full_w for frag in forbidden):
                    continue
            is_paired = bool(r.get("is_paired", r["central"] != r["whorl"]))
            clean.append({
                "subject_id": r.get("subject_id", "?"),
                "central": cp,
                "whorl": wp,
                "is_paired": is_paired,
            })

        if not clean:
            raise RuntimeError(
                "RegionPairDataset received zero valid pairs after whitelist filtering. "
                "Check ALLOWED_PATH_FRAGMENTS / FORBIDDEN_FRAGMENTS and paths."
            )
        self.pairs = clean

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        r = self.pairs[idx]
        c = Image.open(r["central"]).convert("L")
        w = Image.open(r["whorl"]).convert("L")
        if self.transform:
            c, w = self.transform(c), self.transform(w)
        return {
            "x_central": c,
            "x_whorl": w,
            "subject_id": r["subject_id"],
            "is_paired": r["is_paired"],
        }


def assert_no_pd_leak(dataset: RegionPairDataset) -> None:
    """Unit-test helper: raise if any path looks like a PD-positive sample."""
    for r in dataset.pairs:
        for p in (r["central"], r["whorl"]):
            ap = os.path.abspath(p)
            for frag in (os.path.join("pd_dataset_en", "pd"),):
                if frag in ap:
                    raise AssertionError(f"PD leak detected: {ap}")
