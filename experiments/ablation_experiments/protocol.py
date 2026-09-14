"""Small, dependency-free protocol / filesystem helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODES = ROOT.parents[1]
ROUTES = {
    "gt_frangi": ("gt", "frangi"),
    "hgrad_dual_frangi": ("hgrad_dual", "frangi"),
    "frangi_frangi": ("frangi", "frangi"),
    "gt_hgrad_dual": ("gt", "hgrad_dual"),
    "hgrad_dual_hgrad_dual": ("hgrad_dual", "hgrad_dual"),
    "frangi_hgrad_dual": ("frangi", "hgrad_dual"),
}
ENCODER = "topodise_native_zsem"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def lock(path, value):
    if Path(path).exists():
        if read(path) != json.loads(json.dumps(value)):
            raise RuntimeError(f"Locked protocol changed: {path}. Use a new --name.")
    else:
        write(path, value)


def run_root(name):
    if not name or name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("--name must contain only letters, digits, '_' or '-'")
    return ROOT / "results" / name


def source_hashes():
    directories = [CODES / "core", CODES / "topodise", ROOT,
                   CODES / "experiments/topodise_experiments",
                   CODES / "experiments/unified_stage2_encoder_comparison"]
    paths = {p for d in directories for p in d.rglob("*.py")
             if not {"results", "tests", "__pycache__"}.intersection(p.parts)}
    paths.update((CODES / "experiments/_stage_configs.py", CODES / "experiments/__init__.py"))
    return {str(p.relative_to(CODES)): file_hash(p) for p in sorted(paths)}


def checkpoint(run_dir):
    return Path(run_dir) / "ckpt/ema_best.pt"


def require_complete(run_dir):
    run_dir = Path(run_dir)
    done = read(run_dir / "ABLATION_DONE.json")
    path = checkpoint(run_dir)
    if done["checkpoint_sha256"] != file_hash(path):
        raise RuntimeError(f"Completed checkpoint was changed: {path}")
    return path
