from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from core.progress import ETAEstimator

from .encoders.base import FrozenRegionEncoder
from .utils import atomic_json_dump, canonical_hash


def image_morph_stats(image: Image.Image, size: int = 256) -> torch.Tensor:
    gray = TF.to_grayscale(image, num_output_channels=1)
    gray = TF.center_crop(TF.resize(gray, size), [size, size])
    x = TF.to_tensor(gray).float() * 2.0 - 1.0
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    gx_abs, gy_abs = gx.abs(), gy.abs()
    common = torch.sqrt(gx_abs[..., :-1, :] ** 2 + gy_abs[..., :, :-1] ** 2 + 1e-6)
    flat, gradients = x.flatten(), common.flatten()
    mean, std = flat.mean(), flat.std(unbiased=False)
    gmean, gstd = gradients.mean(), gradients.std(unbiased=False)
    return torch.stack([
        mean,
        std,
        (flat > mean + std).float().mean(),
        gmean,
        gstd,
        (gradients > gmean + gstd).float().mean(),
        (gx_abs.mean() - gy_abs.mean()) / (gx_abs.mean() + gy_abs.mean() + 1e-6),
    ])


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


class _ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)
            morphology = image_morph_stats(image)
        return path, tensor, morphology


class _PairPathDataset(Dataset):
    def __init__(self, pairs: list[dict], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        with Image.open(pair["central"]) as handle:
            central_image = handle.convert("RGB")
            central = self.transform(central_image)
            central_morph = image_morph_stats(central_image)
        with Image.open(pair["whorl"]) as handle:
            whorl_image = handle.convert("RGB")
            whorl = self.transform(whorl_image)
            whorl_morph = image_morph_stats(whorl_image)
        morphology = torch.cat([
            central_morph,
            whorl_morph,
            (central_morph - whorl_morph).abs(),
        ])
        return pair["rank_sha256"], central, whorl, morphology


def _extract_region(
    paths: list[str], region: str, encoder: FrozenRegionEncoder, device: torch.device,
    batch_size: int, amp: bool, num_workers: int, pin_memory: bool, channels_last: bool,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    def consume(worker_count: int):
        output: dict[str, torch.Tensor] = {}
        morph: dict[str, torch.Tensor] = {}
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": worker_count,
            "pin_memory": pin_memory,
            "persistent_workers": worker_count > 0,
        }
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = 2
        transform = getattr(encoder, "transform", None)
        if transform is None:
            if worker_count > 0:
                raise RuntimeError(
                    "parallel feature loading requires a standalone encoder.transform; "
                    "a bound CUDA encoder method must not be sent to workers"
                )
            transform = encoder.preprocess
        loader = DataLoader(_ImagePathDataset(paths, transform), **loader_kwargs)
        progress = ETAEstimator(len(paths), window=8)
        completed = 0
        for batch_paths, images, morphology in loader:
            batch_started = time.perf_counter()
            images = images.to(device, non_blocking=pin_memory)
            if channels_last and images.ndim == 4:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), _autocast(device, amp):
                features = encoder.encode_region(images, region).float().cpu()
            if features.ndim != 2 or features.shape[1] != encoder.raw_dim:
                raise RuntimeError(f"{region} encoder output shape {tuple(features.shape)} is invalid")
            output.update({path: feature for path, feature in zip(batch_paths, features)})
            morph.update({path: feature for path, feature in zip(batch_paths, morphology)})
            completed += len(batch_paths)
            state = progress.update(
                completed,
                interval_seconds=time.perf_counter() - batch_started,
            )
            if progress_callback is not None:
                progress_callback({
                    "kind": "feature_progress",
                    "region": region,
                    "worker_count": worker_count,
                    **state,
                })
        return output, morph

    try:
        return consume(num_workers)
    except RuntimeError as error:
        message = str(error)
        shared_memory_denied = "torch_shm_manager" in message and "Operation not permitted" in message
        if num_workers <= 0 or not shared_memory_denied:
            raise
        print("[feature-cache] shared-memory workers unavailable; retrying with num_workers=0")
        return consume(0)


def _extract_pairs(
    pairs: list[dict], encoder: FrozenRegionEncoder, device: torch.device,
    batch_size: int, amp: bool, num_workers: int, pin_memory: bool, channels_last: bool,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    def consume(worker_count: int):
        output: dict[str, torch.Tensor] = {}
        morphology_output: dict[str, torch.Tensor] = {}
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": worker_count,
            "pin_memory": pin_memory,
            "persistent_workers": worker_count > 0,
        }
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = 2
        transform = getattr(encoder, "transform", None)
        if transform is None:
            if worker_count > 0:
                raise RuntimeError(
                    "parallel pair loading requires a standalone encoder.transform"
                )
            transform = encoder.preprocess
        loader = DataLoader(_PairPathDataset(pairs, transform), **loader_kwargs)
        progress = ETAEstimator(len(pairs), window=8)
        completed = 0
        for pair_ids, central, whorl, morphology in loader:
            batch_started = time.perf_counter()
            central = central.to(device, non_blocking=pin_memory)
            whorl = whorl.to(device, non_blocking=pin_memory)
            if channels_last and central.ndim == 4:
                central = central.contiguous(memory_format=torch.channels_last)
                whorl = whorl.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), _autocast(device, amp):
                features = encoder.encode_pair(central, whorl).float().cpu()
            if features.ndim != 2 or features.shape[1] != encoder.raw_dim:
                raise RuntimeError(
                    f"native pair encoder output shape {tuple(features.shape)} is invalid"
                )
            output.update({key: feature for key, feature in zip(pair_ids, features)})
            morphology_output.update({
                key: feature for key, feature in zip(pair_ids, morphology)
            })
            completed += len(pair_ids)
            state = progress.update(
                completed,
                interval_seconds=time.perf_counter() - batch_started,
            )
            if progress_callback is not None:
                progress_callback({
                    "kind": "feature_progress",
                    "region": "native_pair",
                    "worker_count": worker_count,
                    **state,
                })
        return output, morphology_output

    try:
        return consume(num_workers)
    except RuntimeError as error:
        message = str(error)
        shared_memory_denied = (
            "torch_shm_manager" in message and "Operation not permitted" in message
        )
        if num_workers <= 0 or not shared_memory_denied:
            raise
        print("[feature-cache] shared-memory workers unavailable; retrying with num_workers=0")
        return consume(0)


def prepare_feature_cache(
    encoder: FrozenRegionEncoder,
    pair_manifest: dict,
    output_root: Path,
    device: torch.device,
    batch_size: int = 64,
    amp: bool = True,
    num_workers: int = 8,
    pin_memory: bool = False,
    channels_last: bool = True,
    force: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[dict, Path]:
    cache_key = (
        f"{encoder.encoder_name}_{encoder.checkpoint_sha256[:12]}_"
        f"{pair_manifest['manifest_sha256'][:12]}"
    )
    cache_dir = output_root / "feature_cache" / cache_key
    payload_path = cache_dir / "features.pt"
    meta_path = cache_dir / "metadata.json"
    if payload_path.is_file() and meta_path.is_file() and not force:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        cache_matches = (
            metadata.get("encoder") == encoder.encoder_name
            and metadata.get("checkpoint_sha256") == encoder.checkpoint_sha256
            and metadata.get("pair_manifest_sha256") == pair_manifest["manifest_sha256"]
            and metadata.get("feature_mode") == encoder.feature_mode
        )
        if cache_matches:
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            expected_keys = (
                {"pair", "pair_morphology"}
                if encoder.feature_mode == "native_pair"
                else {"central", "whorl", "morph"}
            )
            if set(payload) == expected_keys:
                if progress_callback is not None:
                    progress_callback({
                        "kind": "feature_cache_hit",
                        "cache_dir": str(cache_dir),
                    })
                return payload, cache_dir

    encoder.to(device).freeze().assert_frozen()
    if channels_last:
        encoder.to(memory_format=torch.channels_last)
    if encoder.feature_mode == "native_pair":
        selected_pairs = [
            pair
            for patient in pair_manifest["patients"]
            for pair in patient["selected_pairs"]
        ]
        pair_ids = [pair["rank_sha256"] for pair in selected_pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise RuntimeError("pair manifest contains duplicate rank_sha256 identifiers")
        pair_features, pair_morphology = _extract_pairs(
            selected_pairs, encoder, device, batch_size, amp,
            num_workers, pin_memory, channels_last, progress_callback,
        )
        payload = {"pair": pair_features, "pair_morphology": pair_morphology}
        payload_contract = {
            "pair": sorted(pair_features),
            "pair_morphology": sorted(pair_morphology),
        }
        count_metadata = {"num_pairs": len(pair_features)}
    elif encoder.feature_mode == "region":
        central_paths = sorted({
            pair["central"]
            for patient in pair_manifest["patients"]
            for pair in patient["selected_pairs"]
        })
        whorl_paths = sorted({
            pair["whorl"]
            for patient in pair_manifest["patients"]
            for pair in patient["selected_pairs"]
        })
        central, morph_c = _extract_region(
            central_paths, "central", encoder, device, batch_size, amp,
            num_workers, pin_memory, channels_last, progress_callback,
        )
        whorl, morph_w = _extract_region(
            whorl_paths, "whorl", encoder, device, batch_size, amp,
            num_workers, pin_memory, channels_last, progress_callback,
        )
        morph = {**morph_c, **morph_w}
        payload = {"central": central, "whorl": whorl, "morph": morph}
        payload_contract = {
            "central": sorted(central),
            "whorl": sorted(whorl),
            "morph": sorted(morph),
        }
        count_metadata = {
            "num_central_images": len(central),
            "num_whorl_images": len(whorl),
        }
    else:
        raise ValueError(f"unsupported encoder feature_mode: {encoder.feature_mode}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = payload_path.with_suffix(f".tmp.{os.getpid()}.pt")
    torch.save(payload, tmp)
    os.replace(tmp, payload_path)
    metadata = {
        **encoder.metadata(),
        "pair_manifest_sha256": pair_manifest["manifest_sha256"],
        **count_metadata,
        "loader": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "channels_last": channels_last,
            "morphology_fused_with_decode": True,
        },
        "payload_sha256_contract": canonical_hash(payload_contract),
    }
    atomic_json_dump(metadata, meta_path)
    return payload, cache_dir


def build_patient_bags(pair_manifest: dict, feature_payload: dict, labels: dict[str, int]) -> dict[str, dict]:
    bags = {}
    for patient in pair_manifest["patients"]:
        pid = patient["patient_id"]
        if pid not in labels:
            continue
        pair_morph = []
        if "pair" in feature_payload:
            pair_features = []
            for pair in patient["selected_pairs"]:
                pair_id = pair["rank_sha256"]
                pair_features.append(feature_payload["pair"][pair_id])
                pair_morph.append(feature_payload["pair_morphology"][pair_id])
            feature_fields = {"pair_features": torch.stack(pair_features)}
            count = len(pair_features)
        else:
            central, whorl = [], []
            for pair in patient["selected_pairs"]:
                c, w = pair["central"], pair["whorl"]
                central.append(feature_payload["central"][c])
                whorl.append(feature_payload["whorl"][w])
                cm, wm = feature_payload["morph"][c], feature_payload["morph"][w]
                pair_morph.append(torch.cat([cm, wm, (cm - wm).abs()]))
            feature_fields = {
                "central": torch.stack(central),
                "whorl": torch.stack(whorl),
            }
            count = len(central)
        if count == 0:
            raise RuntimeError(f"patient {pid} has an empty pair bag")
        bags[pid] = {
            "patient_id": pid,
            "label": int(labels[pid]),
            "morphology": torch.stack(pair_morph).mean(dim=0),
            **feature_fields,
        }
    return bags
