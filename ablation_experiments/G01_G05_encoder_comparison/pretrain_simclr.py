from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from core.simclr_trainer import SimCLRModel, simclr_pretrain
from topodise.data import RegionPairDataset

from .config import load_config
from .utils import append_jsonl, canonical_hash, pick_device, resolve_codes_path, seed_everything


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class UnlabeledImageDataset:
    samples: list[str]
    mode: str = "single_stream"

    def __len__(self) -> int:
        return len(self.samples)


def _pd_image_paths(dataset_json: Path, dataset_root: Path) -> list[Path]:
    with dataset_json.open("r", encoding="utf-8") as handle:
        patients = json.load(handle)
    paths: set[Path] = set()
    for patient in patients:
        image_paths = patient.get("image_paths") or {}
        for region in ("central", "whorl"):
            for relative in image_paths.get(region, []):
                path = (dataset_root / relative).resolve()
                if path.is_file():
                    paths.add(path)
    return sorted(paths)


def _directory_image_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"SimCLR pretraining directory does not exist: {root}")
    return sorted(p.resolve() for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _corn1_image_paths(root: Path) -> list[Path]:
    image_root = root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"CORN-1 images directory does not exist: {image_root}")
    return _directory_image_paths(image_root)


def _region_manifest_image_paths(manifest: Path, dataset_root: Path) -> tuple[list[Path], dict]:
    dataset = RegionPairDataset(
        manifest_path=str(manifest),
        base_path=str(dataset_root),
        transform=None,
        strict_whitelist=True,
        allow_dm_disease=False,
    )
    paths = sorted({Path(row[key]).resolve() for row in dataset.pairs for key in ("central", "whorl")})
    paired = sum(bool(row["is_paired"]) for row in dataset.pairs)
    return paths, {
        "num_region_rows": len(dataset.pairs),
        "num_paired_region_rows": paired,
        "num_unpaired_region_rows": len(dataset.pairs) - paired,
        "skipped_duplicate_filenames": dataset.skipped_duplicate_filenames,
    }


def collect_pretrain_images(spec: dict) -> tuple[list[Path], dict]:
    source = str(spec.get("source", "pd_all_unlabeled"))
    if source == "pd_all_unlabeled":
        dataset_json = resolve_codes_path(spec.get("dataset_json", "datasets/pd_dataset_en/patients_dataset_en.json"))
        dataset_root = resolve_codes_path(spec.get("dataset_root", "datasets"))
        paths = _pd_image_paths(dataset_json, dataset_root)
        source_meta = {
            "source": source,
            "dataset_json": str(dataset_json),
            "dataset_root": str(dataset_root),
            "transductive_unlabeled_pretraining": True,
        }
        stable_ids = [str(path.relative_to(dataset_root)) for path in paths]
    elif source == "directory":
        source_root = resolve_codes_path(spec["source_root"])
        paths = _directory_image_paths(source_root)
        source_meta = {
            "source": source,
            "source_root": str(source_root),
            "transductive_unlabeled_pretraining": False,
        }
        stable_ids = [str(path.relative_to(source_root)) for path in paths]
    elif source == "corn1_raw":
        source_root = resolve_codes_path(spec.get("source_root", "datasets/CORN-1"))
        paths = _corn1_image_paths(source_root)
        source_meta = {
            "source": source,
            "source_root": str(source_root),
            "uses_segmentation_masks": False,
            "transductive_unlabeled_pretraining": False,
        }
        stable_ids = [str(path.relative_to(source_root)) for path in paths]
    elif source == "region_manifest":
        manifest = resolve_codes_path(spec["manifest"])
        dataset_root = resolve_codes_path(spec.get("dataset_root", "datasets"))
        paths, region_meta = _region_manifest_image_paths(manifest, dataset_root)
        source_meta = {
            "source": source,
            "manifest": str(manifest),
            "dataset_root": str(dataset_root),
            "transductive_unlabeled_pretraining": False,
            **region_meta,
        }
        stable_ids = [str(path.relative_to(dataset_root)) for path in paths]
    else:
        raise ValueError(f"unsupported SimCLR pretrain source: {source}")
    if len(paths) < 2:
        raise RuntimeError(f"SimCLR pretraining needs at least two readable images; found {len(paths)}")
    source_meta["num_images"] = len(paths)
    source_meta["image_manifest_sha256"] = canonical_hash(stable_ids)
    return paths, source_meta


def _save_torch_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _stage_cache_path(checkpoint: Path, index: int, stage_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stage_name)
    return checkpoint.with_name(f"{checkpoint.stem}.stage{index + 1}_{safe_name}.pt")


def pretrain_and_cache(
    checkpoint: Path,
    spec: dict,
    device_request: str = "auto",
    reuse_stage_cache: bool = True,
) -> Path:
    checkpoint = checkpoint.resolve()
    seed = int(spec.get("seed", 20260831))
    seed_everything(seed)
    device = pick_device(device_request)
    initialization = spec.get("initialization", False)
    if initialization not in (False, True, "Half"):
        raise ValueError("SimCLR initialization must be false, true, or 'Half'")
    stages = spec.get("stages") or [spec]
    progress_path = checkpoint.with_name(f"{checkpoint.stem}_progress.jsonl")

    def emit_progress(event: dict) -> None:
        append_jsonl({"wall_time": time.time(), **event}, progress_path)

    emit_progress({
        "kind": "pipeline_start",
        "method": "simclr",
        "checkpoint": str(checkpoint),
        "num_stages": len(stages),
    })
    model = None
    stage_records = []
    for index, stage in enumerate(stages):
        stage_spec = {**spec, **stage}
        stage_spec.pop("stages", None)
        stage_name = str(stage.get("name", f"stage_{index + 1}"))
        paths, source_meta = collect_pretrain_images(stage_spec)
        stage_cache = _stage_cache_path(checkpoint, index, stage_name)
        stage_spec_sha256 = canonical_hash(stage_spec)
        cache_usable = False
        cached = None
        if index < len(stages) - 1 and reuse_stage_cache and stage_cache.is_file():
            cached = torch.load(stage_cache, map_location="cpu", weights_only=False)
            cache_usable = (
                cached.get("stage_spec_sha256") == stage_spec_sha256
                and cached.get("source", {}).get("image_manifest_sha256")
                == source_meta["image_manifest_sha256"]
            )
            if not cache_usable:
                print(f"[SimCLR] ignoring stale stage cache: {stage_cache}")
        if cache_usable:
            model = SimCLRModel(
                "resnet18",
                pretrained=False,
                proj_hidden_dim=int(spec.get("proj_hidden_dim", 256)),
                proj_out_dim=int(spec.get("proj_out_dim", 128)),
            )
            model.load_state_dict(cached["model_state"])
            print(f"[SimCLR] resume from completed {stage_name} cache: {stage_cache}")
            emit_progress({
                "kind": "stage_cache_hit",
                "stage": stage_name,
                "stage_index": index + 1,
                "stage_count": len(stages),
                "cache": str(stage_cache),
            })
        else:
            print(
                f"[SimCLR] {stage_name}: {len(paths)} unlabeled images, "
                f"{int(stage_spec.get('epochs', 100))} epochs on {device}"
            )
            dataset = UnlabeledImageDataset([str(path) for path in paths])
            stage_started = time.perf_counter()
            emit_progress({
                "kind": "stage_start",
                "stage": stage_name,
                "stage_index": index + 1,
                "stage_count": len(stages),
                "num_images": len(paths),
                "epochs": int(stage_spec.get("epochs", 100)),
            })
            model = simclr_pretrain(
                backbone_name="resnet18",
                pretrained=initialization if model is None else False,
                train_dataset=dataset,
                num_epochs=int(stage_spec.get("epochs", 100)),
                batch_size=int(stage_spec.get("batch_size", 32)),
                lr=float(stage_spec.get("lr", 1e-3)),
                temperature=float(stage_spec.get("temperature", 0.5)),
                device=str(device),
                proj_hidden_dim=int(spec.get("proj_hidden_dim", 256)),
                proj_out_dim=int(spec.get("proj_out_dim", 128)),
                weight_decay=float(stage_spec.get("weight_decay", 1e-4)),
                num_workers=int(stage_spec.get("num_workers", 8)),
                input_size=224,
                verbose=True,
                amp=bool(stage_spec.get("amp", True)),
                channels_last=bool(stage_spec.get("channels_last", True)),
                persistent_workers=bool(stage_spec.get("persistent_workers", True)),
                pin_memory=bool(stage_spec.get("pin_memory", False)),
                fused_adamw=bool(stage_spec.get("fused_adamw", True)),
                drop_last=bool(stage_spec.get("drop_last", True)),
                initial_model=model,
                progress_callback=emit_progress,
                progress_stage=stage_name,
            )
            emit_progress({
                "kind": "stage_complete",
                "stage": stage_name,
                "stage_index": index + 1,
                "stage_count": len(stages),
                "elapsed_seconds": round(time.perf_counter() - stage_started, 3),
            })
            if index < len(stages) - 1:
                _save_torch_atomic(
                    {
                        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "stage_name": stage_name,
                        "source": source_meta,
                        "stage_spec_sha256": stage_spec_sha256,
                    },
                    stage_cache,
                )
        stage_records.append({
            "name": stage_name,
            "source": source_meta,
            "epochs": int(stage_spec.get("epochs", 100)),
            "batch_size": int(stage_spec.get("batch_size", 32)),
            "lr": float(stage_spec.get("lr", 1e-3)),
        })
    if model is None:
        raise RuntimeError("SimCLR pretraining pipeline contains no stages")
    pretrain_hparams = {key: spec.get(key) for key in (
        "initialization", "temperature", "proj_hidden_dim", "proj_out_dim",
        "weight_decay", "seed", "amp", "channels_last", "persistent_workers",
        "pin_memory", "fused_adamw", "drop_last",
    )}
    payload = {
        "backbone_state": {key: value.detach().cpu() for key, value in model.get_backbone().state_dict().items()},
        "meta": {
            "backbone": "resnet18",
            "method": "simclr_nt_xent",
            "pipeline": stage_records,
            "pretrain_hparams": pretrain_hparams,
            "pretrain_spec_sha256": canonical_hash(spec),
        },
    }
    _save_torch_atomic(payload, checkpoint)
    emit_progress({
        "kind": "pipeline_complete",
        "method": "simclr",
        "checkpoint": str(checkpoint),
    })
    return checkpoint


def ensure_simclr_checkpoint(checkpoint: Path, encoder_spec: dict, device_request: str = "auto") -> Path:
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        observed = payload.get("meta", {}).get("pretrain_spec_sha256")
        expected = canonical_hash(encoder_spec.get("pretrain", {}))
        if observed is not None and observed != expected:
            raise RuntimeError(
                "existing SimCLR cache was trained with a different pretraining protocol; "
                f"move it aside or update checkpoint path: {checkpoint}"
            )
        return checkpoint
    if not bool(encoder_spec.get("auto_pretrain_if_missing", True)):
        raise FileNotFoundError(f"SimCLR checkpoint not found and auto-pretrain is disabled: {checkpoint}")
    return pretrain_and_cache(checkpoint, encoder_spec.get("pretrain", {}), device_request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain/cache the SimCLR ResNet-18 comparator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    spec = cfg["encoders"]["simclr_resnet18"]
    checkpoint = resolve_codes_path(spec["checkpoint"])
    if checkpoint.exists() and not args.force:
        ensure_simclr_checkpoint(checkpoint, spec, args.device)
        print(f"[SimCLR] using compatible existing cache: {checkpoint}")
        return
    pretrain_and_cache(
        checkpoint,
        spec.get("pretrain", {}),
        args.device,
        reuse_stage_cache=not args.force,
    )


if __name__ == "__main__":
    main()
