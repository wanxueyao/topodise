"""Stage 2 (RACF): residual-adapter classifier finetune.

PD has three sub-tasks (pd_binary / pd_stage / pd_subtype).
DM has two sub-tasks (dm_binary: control vs DM; dm_subtype: NDR vs NPDR).
"""


import os
import sys
import json
import time
import random
import copy
import shutil
import math
import contextlib
import hashlib
from dataclasses import asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

from topodise import (
    TopoDiSEEncoder, EncoderConfig,
    ConditionalUNet, UNetConfig,
    TopoHead, DiffusionSchedule, q_sample,
    FeatureDiffusionHead, q_sample_feature,
    apply_freeze_policy, attach_adapters,
    ClassifierHead, ProjectionHead, CoralHead, supcon_loss, focal_ce_loss,
    coral_loss, coral_predict,
    PrototypeHead, episodic_proto_loss,
    TrainLogger, EMAMeter, RateMeter, count_params, grad_global_norm, env_banner,
)
from topodise.data import (
    DownstreamDataset, PD_TASKS,
    build_pretrain_transform, build_eval_transform,
)
from topodise.data.downstream_dataset import _load_gray, _GRAY_IMAGE_CACHE
from topodise.data.dm_dataset import DMDownstreamDataset, DM_TASKS
from core.data_loader import stratified_kfold
from .configs import Stage2Config
from .hooks import current_hooks


def _cap_cpu_threads():
    """Optionally cap intra-process CPU threads.

    When several Stage 2 runs share one GPU (RUN_CONCURRENCY_PER_GPU>1), each
    process otherwise grabs every core for BLAS/transform work and the runs
    thrash the CPU instead of feeding the GPU. Set TOPODISE_NUM_THREADS=N to cap
    each process. Off by default to preserve single-run behaviour.
    """
    n = os.environ.get("TOPODISE_NUM_THREADS", "").strip()
    if not n:
        return
    try:
        k = max(1, int(n))
        torch.set_num_threads(k)
        try:
            torch.set_num_interop_threads(max(1, k // 2))
        except RuntimeError:
            pass  # 进程间线程池已初始化，忽略此异常。
    except ValueError:
        pass


_cap_cpu_threads()


# 编码器前向计算使用 bf16 自动混合精度。编码器卷积/UNet 的前向计算
# （加上编码器内适配器的反向计算）占 Stage2 的主要时间；在 Ampere 及更新 GPU 上，
# bf16 通常可带来约 1.5–2 倍加速，且无需 GradScaler（bf16 的取值范围接近 fp32）。
# 仅编码器前向计算位于自动混合精度上下文中；其输出会转回 fp32，
# 使小型分类头和对数值敏感的损失（supcon/coral/proto/focal，
# 其中使用 exp/log）保持 fp32。CUDA 默认开启，可用 TOPODISE_AMP=0 禁用；
# 在 CPU/MPS 上不执行任何额外操作。
_AMP_ENABLED = os.environ.get("TOPODISE_AMP", "1").strip() not in ("0", "false", "False")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _configure_reproducibility(cfg: Stage2Config) -> None:
    """Resolve and apply the Stage2 reproducibility contract before CUDA use."""
    cfg.deterministic = bool(getattr(cfg, "deterministic", False)) or _env_flag(
        "TOPODISE_DETERMINISTIC", False
    )
    seed = int(cfg.seed)
    # Preserve the historical portable Stage2 method, which seeded only Torch.
    torch.manual_seed(seed)
    if cfg.deterministic:
        # This must be present before the first CUDA call. Launchers should set
        # it before Python starts; setdefault protects direct entry points.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def _autocast(device):
    dev_type = device.type if hasattr(device, "type") else str(device)
    if _AMP_ENABLED and dev_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --- 单任务与多任务运行器共用辅助函数 -------------------------------

def _compute_log_prior(labels, n_classes, device):
    counts = [0] * n_classes
    for y in labels:
        if 0 <= y < n_classes:
            counts[y] += 1
    total = max(sum(counts), 1)
    prior = [max(c, 1) / total for c in counts]
    return torch.log(torch.tensor(prior, dtype=torch.float32, device=device))


def _minority_mask(y, class_counts):
    """Return float mask of shape [B], 1.0 for samples whose class is
    among the bottom-half of class frequencies. ``class_counts`` is a dict or
    list mapping class -> count."""
    if isinstance(class_counts, dict):
        sorted_cls = sorted(class_counts.keys(), key=lambda c: class_counts[c])
    else:
        sorted_cls = sorted(range(len(class_counts)), key=lambda c: class_counts[c])
    n_min = max(1, len(sorted_cls) // 2)
    minority = set(sorted_cls[:n_min])
    return torch.tensor([1.0 if int(yi) in minority else 0.0 for yi in y.tolist()],
                        device=y.device, dtype=torch.float32)


def _ce_or_focal(logits, y, cfg, weight):
    """Cross-entropy or focal-CE depending on cfg.use_focal."""
    if getattr(cfg, "use_focal", False):
        return focal_ce_loss(logits, y, gamma=cfg.focal_gamma,
                             weight=weight, label_smoothing=cfg.label_smooth_eps)
    return F.cross_entropy(logits, y, weight=weight,
                           label_smoothing=cfg.label_smooth_eps)


def _apply_class_weight_multipliers(class_weight_t, cfg):
    multipliers = tuple(getattr(cfg, "class_weight_multipliers", ()) or ())
    if class_weight_t is None or not multipliers:
        return class_weight_t
    for idx, mult in enumerate(multipliers[:class_weight_t.numel()]):
        class_weight_t[idx] = class_weight_t[idx] * float(mult)
    return class_weight_t


def _binary_threshold_predict(logits: torch.Tensor, cfg, n_classes: int) -> torch.Tensor:
    cls = int(getattr(cfg, "binary_threshold_class", -1))
    if n_classes != 2 or cls < 0 or cls > 1:
        return logits.argmax(-1)
    prob = torch.softmax(logits, dim=-1)[..., cls]
    other = 1 - cls
    return torch.where(
        prob >= float(getattr(cfg, "binary_threshold", 0.5)),
        torch.full_like(prob, cls, dtype=torch.long),
        torch.full_like(prob, other, dtype=torch.long),
    )


class _WeightEMA:
    """Exponential moving average over a fixed list of parameters.

    ``store_and_copy`` swaps the shadow (EMA) weights into the live params for
    evaluation/checkpointing; ``restore`` puts the training weights back.
    """

    def __init__(self, params, decay: float = 0.999):
        self.decay = decay
        self.params = list(params)
        self.shadow = [p.detach().clone() for p in self.params]
        self._backup = None

    @torch.no_grad()
    def update(self):
        d = self.decay
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def store_and_copy(self):
        self._backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)

    @torch.no_grad()
    def restore(self):
        if self._backup is None:
            return
        for p, b in zip(self.params, self._backup):
            p.data.copy_(b)
        self._backup = None






def _stage2_resume_key(cfg: Stage2Config) -> str:
    """Stable key for fold-level Stage2 resume within one run directory."""
    d = asdict(cfg)
    # 这些字段只描述输出位置，不属于实验语义。
    d.pop("output_dir", None)
    d.pop("run_name", None)
    payload = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _partial_results_path(run_dir: str) -> str:
    return os.path.join(run_dir, "partial_results.json")


def _load_partial_fold_results(run_dir: str, cfg: Stage2Config, n_splits: int, logger) -> List[dict]:
    path = _partial_results_path(run_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.info(f"[resume][stage2] ignore unreadable partial results: {e}")
        return []
    if payload.get("resume_key") != _stage2_resume_key(cfg):
        logger.info("[resume][stage2] ignore partial results: config key mismatch")
        return []
    if int(payload.get("evaluated_folds", -1)) != int(n_splits):
        logger.info("[resume][stage2] ignore partial results: fold-count mismatch")
        return []
    out = []
    seen = set()
    ckpt_dir = os.path.join(run_dir, "ckpt")
    for r in payload.get("fold_results", []):
        fold = int(r.get("fold", -1))
        if fold < 1 or fold > n_splits or fold in seen:
            continue
        ckpt = os.path.join(ckpt_dir, f"best_fold{fold}.pt")
        if not os.path.exists(ckpt):
            logger.info(f"[resume][stage2] fold {fold} partial ignored: missing {ckpt}")
            continue
        out.append(r)
        seen.add(fold)
    out.sort(key=lambda x: int(x["fold"]))
    if out:
        logger.info(f"[resume][stage2] loaded completed folds: {[r['fold'] for r in out]}")
    return out


def _save_partial_fold_results(run_dir: str, cfg: Stage2Config, n_splits: int, fold_results: List[dict]) -> None:
    path = _partial_results_path(run_dir)
    tmp = path + ".tmp"
    payload = {
        "resume_key": _stage2_resume_key(cfg),
        "task": cfg.task,
        "evaluated_folds": n_splits,
        "fold_results": sorted(fold_results, key=lambda x: int(x["fold"])),
        "updated": time.time(),
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    _update_batch_stage2_partial_summary(cfg, n_splits, fold_results)


def _update_batch_stage2_partial_summary(cfg: Stage2Config, n_splits: int,
                                         fold_results: List[dict]) -> None:
    """Mirror fold-level progress into the batch summary when run under batch_experiments."""
    summary_path = os.path.join(cfg.output_dir, "summary.json")
    if not os.path.exists(summary_path):
        return
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    stage2 = summary.setdefault("stage2_results", {})
    previous = stage2.get(cfg.task)
    if not isinstance(previous, dict):
        previous = {}
    previous.update({
        "status": "running",
        "task": cfg.task,
        "partial": True,
        "completed_folds": len(fold_results),
        "num_folds": cfg.num_folds,
        "eval_folds": cfg.eval_folds,
        "evaluated_folds": n_splits,
        "fold_results": sorted(fold_results, key=lambda x: int(x["fold"])),
        "updated": time.time(),
    })
    stage2[cfg.task] = previous
    tmp = summary_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        os.replace(tmp, summary_path)
    except OSError:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass




def _fdiff_anchor(z: torch.Tensor, grad_scale: float) -> torch.Tensor:
    """Return a source anchor with configurable gradient flow.

    Keeps the forward value equal to ``z`` while scaling the backward signal
    through the anchor path. ``grad_scale=0`` recovers the old stop-grad path.
    """
    grad_scale = float(max(0.0, min(1.0, grad_scale)))
    if grad_scale <= 0:
        return z.detach()
    if grad_scale >= 1:
        return z
    return z.detach() + grad_scale * (z - z.detach())


def _fdiff_gate(z_aug: torch.Tensor, z_ref: torch.Tensor, cfg) -> torch.Tensor:
    """Gate or blend augmented features back toward the source feature."""
    mode = getattr(cfg, "fdiff_gate_mode", "hard_cos")
    if mode == "none":
        return z_aug
    cos = F.cosine_similarity(z_aug, z_ref.detach(), dim=-1).unsqueeze(-1)
    thr = float(getattr(cfg, "fdiff_gate_threshold", getattr(cfg, "fdiff_min_cos", 0.85)))
    if mode == "soft_cos":
        temp = max(float(getattr(cfg, "fdiff_gate_temperature", 0.10)), 1e-6)
        trust = torch.sigmoid((cos - thr) / temp)
    else:
        trust = (cos > thr).float()
    return trust * z_aug + (1.0 - trust) * z_ref


def _feature_aug_step_mil(z, y, pool, cls_head, fdiff_head, schedule, cfg,
                          n_classes, class_counts, class_weight, ordinal,
                          morph_token=None, stage_thresholds=False,
                          threshold_pos_weight=None):
    """Feature diffusion augmentation for a single MIL patient bag.

    Builds augmented *instance* features, then re-pools them to a bag embedding
    and applies the bag-level supervision. This keeps the augmentation aligned
    with the MIL training objective instead of image-pair CE.
    """
    device = z.device
    n_inst = z.size(0)
    T_aug = max(1, min(cfg.fdiff_T_aug, schedule.T - 1))
    y_vec = torch.full((n_inst,), int(y), device=device, dtype=torch.long)
    z_anchor = _fdiff_anchor(z, getattr(cfg, "fdiff_target_grad_scale", 0.0))

    t = torch.randint(1, T_aug + 1, (n_inst,), device=device)
    eps_n = torch.randn_like(z_anchor)
    z_t = q_sample_feature(z_anchor, t, eps_n, schedule.alpha_bar)
    z0_hat = fdiff_head(z_t, t, y_vec if cfg.fdiff_class_cond else None)
    L_fdiff = F.mse_loss(z0_hat, z_anchor)

    L_ce_aug = torch.zeros((), device=device)
    if cfg.lambda_ce_aug > 0 and cfg.fdiff_k_aug > 0:
        if cfg.fdiff_minority_only and class_counts:
            count_y = class_counts.get(int(y), 0)
            if count_y > min(class_counts.values() or [count_y]):
                return L_fdiff, L_ce_aug
        total = torch.zeros((), device=device)
        for _ in range(cfg.fdiff_k_aug):
            t2 = torch.randint(1, T_aug + 1, (n_inst,), device=device)
            eps2 = torch.randn_like(z_anchor)
            z_t2 = q_sample_feature(z_anchor, t2, eps2, schedule.alpha_bar)
            z_aug_raw = fdiff_head(z_t2, t2, y_vec if cfg.fdiff_class_cond else None)
            z_aug = _fdiff_gate(z_aug_raw, z_anchor, cfg)
            current_hooks().feature_diagnostics(z_aug_raw, z_anchor, z_aug, y_vec, cfg, "training")
            m_aug, _ = pool(z_aug)
            if morph_token is not None:
                m_aug = torch.cat([m_aug, morph_token], dim=1)
            logits_aug = cls_head(m_aug)
            if stage_thresholds:
                per = _stage_threshold_loss(logits_aug,
                                            torch.tensor([int(y)], device=device),
                                            pos_weight=threshold_pos_weight)
            elif ordinal:
                per = coral_loss(logits_aug, torch.tensor([int(y)], device=device),
                                 n_classes, weight=class_weight)
            else:
                per = _ce_or_focal(logits_aug,
                                   torch.tensor([int(y)], device=device),
                                   cfg, class_weight)
            total = total + per
        L_ce_aug = total / cfg.fdiff_k_aug
    return L_fdiff, L_ce_aug


# --- 测试时增强（保持标签的几何视图） ------------------------------

def _tta_views(x, n_views: int):
    """Return up to ``n_views`` label-preserving views of a [B,C,H,W] tensor:
    identity, hflip, vflip, rot180. Fundus image-pairs are orientation-agnostic
    for PD/stage/subtype labels, so these views preserve the target."""
    views = [x]
    if n_views >= 2:
        views.append(torch.flip(x, dims=[-1]))      # 水平翻转。
    if n_views >= 3:
        views.append(torch.flip(x, dims=[-2]))      # 垂直翻转。
    if n_views >= 4:
        views.append(torch.flip(x, dims=[-1, -2]))  # 旋转 180 度。
    return views[:max(1, n_views)]


@torch.no_grad()
def _tta_logits(encoder, cls_head, xc, xw, n_views: int):
    """Average softmax over TTA views and return an equivalent logit tensor
    (log of the mean probability), so downstream argmax and patient-level
    aggregation operate on the same calibrated scale as the single-view path."""
    vc = _tta_views(xc, n_views)
    vw = _tta_views(xw, n_views)
    prob_sum = None
    for cc, ww in zip(vc, vw):
        with _autocast(xc.device):
            z, _ = encoder(cc, ww)
        p = torch.softmax(cls_head(z.float()), dim=-1)
        prob_sum = p if prob_sum is None else prob_sum + p
    prob_mean = prob_sum / len(vc)
    return torch.log(prob_mean.clamp_min(1e-8))


class _GatedAttentionPool(nn.Module):
    """Gated-attention MIL pooling (Ilse et al., 2018, "Attention-based Deep
    Multiple Instance Learning"). Pools a bag of instance embeddings
    ``H = [N, D]`` into a single bag embedding via learned attention weights.
    """

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, H):
        # H: [N, D]；在 N 个实例上计算注意力。
        a = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))  # 形状：[N, 1]
        a = torch.softmax(a, dim=0)
        m = (a * H).sum(dim=0, keepdim=True)                          # 形状：[1, D]
        return self.drop(m), a.squeeze(-1)


class _MeanMILPool(nn.Module):
    """Mean-pooling MIL baseline with the same output contract as attention."""

    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, H):
        a = torch.full((H.size(0),), 1.0 / max(H.size(0), 1), device=H.device, dtype=H.dtype)
        return self.drop(H.mean(dim=0, keepdim=True)), a


NUM_CLASSES = {
    "pd_binary": 2, "pd_stage": 3, "pd_subtype": 3,
    "pd_subtype_td_pigd": 2, "pd_subtype_tdrest": 2, "pd_subtype_pigd_mixed": 2,
    "pd_stage_ge2": 2, "pd_stage_ge3": 2,
    "dm_binary": 2, "dm_subtype": 2, "dm_threeway": 3,
}


def _pick_device(dev):
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


def _build_from_ckpt(ckpt_path, device, cfg=None):
    if cfg is not None and cfg.encoder_backend != "topodise":
        return current_hooks().build_external_encoder(cfg, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pc = ckpt.get("parent_config") or ckpt["config"]
    enc_cfg = EncoderConfig(
        in_channels=1,
        base_channels=pc["encoder_base_channels"],
        channel_mult=tuple(pc["encoder_channel_mult"]),
        blocks_per_stage=tuple(pc["encoder_blocks_per_stage"]),
        z_dim=pc["z_dim"],
    )
    unet_cfg = UNetConfig(
        in_channels=1, out_channels=1,
        model_channels=pc["unet_model_channels"],
        channel_mult=tuple(pc["unet_channel_mult"]),
        z_dim=pc["z_dim"], image_size=pc["image_size"],
    )
    encoder = current_hooks().build_encoder(enc_cfg, pc).to(device)
    unet = ConditionalUNet(unet_cfg).to(device)
    topo_head = TopoHead(z_dim=pc["z_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    unet.load_state_dict(ckpt["unet"])
    topo_head.load_state_dict(ckpt["topo_head"])
    return encoder, unet, topo_head, pc


def _train_val_split(ds, val_ratio: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    return indices[n_val:], indices[:n_val]


def _clone_dataset_with_transform(ds, transform, cache_final=True):
    clone = copy.copy(ds)
    clone.transform = transform
        # 评估变换是确定性的，因此可按索引缓存完整变换后的张量。
        # 评估/原型循环每轮都会重新读取所有验证（以及训练）包；
        # 缓存可将重复的解码和变换转为字典查找。若变换不具确定性，
        # 则通过 cache_final 禁用该缓存。
    clone._final_cache = {} if cache_final else None
    return clone


def _warm_image_cache(ds, logger=None, tag: str = ""):
    """Decode every unique image referenced by `ds` once, populating the
    process-global decode cache before training starts.

    The runners decode images lazily on first access, so the very first
    epoch/fold otherwise silently pays the full disk-decode cost with no log
    output (a multi-minute gap after "trainable total"). Warming the cache up
    front turns that hidden gap into an explicit, logged phase and guarantees
    every subsequent fold/epoch is a cache hit. The MIL runner decodes bags in
    the main process, so this directly removes its startup stall.
    """
    samples = getattr(ds, "samples", None)
    if not samples:
        return
    paths = set()
    for s in samples:
        if len(s) >= 2 and isinstance(s[0], str) and isinstance(s[1], str):
            paths.add(s[0])
            paths.add(s[1])
    if not paths:
        return
    t0 = time.time()
    if logger is not None:
        logger.info(f"warming image cache{tag}: decoding {len(paths)} unique images ...")
    decoded = 0
    for p in paths:
        try:
            _load_gray(p)
            decoded += 1
        except Exception as e:  # noqa: BLE001 - 单个错误路径不应中断训练。
            if logger is not None:
                logger.info(f"  [warm] skip {p}: {e}")
    if logger is not None:
        logger.info(f"image cache ready{tag}: {decoded}/{len(paths)} images decoded "
                    f"in {time.time() - t0:.1f}s (cache size={len(_GRAY_IMAGE_CACHE)})")


def _build_kfold_splits(ds, cfg: Stage2Config):
    explicit = current_hooks().split_indices(ds, cfg)
    if explicit is not None:
        return explicit
    if cfg.num_folds < 2:
        tr, va = _train_val_split(ds, cfg.val_split, cfg.seed)
        return [(tr, va)]

    labels = [int(s[2]) for s in ds.samples]
    patient_ids = [s[3] for s in ds.samples]
    repeats = max(1, getattr(cfg, "kfold_repeats", 1))
    all_splits = []
    for rep in range(repeats):
        splits = list(stratified_kfold(
            labels, labels, labels,
            n_splits=cfg.num_folds,
            random_state=cfg.seed + 1000 * rep,
            kfold_mode=cfg.kfold_mode,
            patient_ids=patient_ids,
        ))
        if cfg.eval_folds > 0:
            splits = splits[:min(cfg.eval_folds, len(splits))]
        all_splits.extend((list(tr), list(va)) for tr, va in splits)
    return all_splits


def _make_dataset(cfg: Stage2Config, transform):
    if cfg.task in PD_TASKS:
        dataset = DownstreamDataset(cfg.pd_json, cfg.base_path, task=cfg.task,
                                  transform=transform,
                                  pair_mode=cfg.pair_mode,
                                  max_pairs_per_patient=cfg.max_pairs_per_patient,
                                  seed=cfg.seed)
    elif cfg.task in DM_TASKS:
        dataset = DMDownstreamDataset(cfg.pd_json, cfg.base_path, task=cfg.task,
                                    transform=transform, pair_mode=cfg.dm_pair_mode)
    else:
        raise ValueError(f"unknown task: {cfg.task}")
    return current_hooks().dataset(dataset, cfg)




def run_single_experiment(cfg: Stage2Config) -> dict:
    """Current method uses patient MIL; research policies use the same trainer."""
    if cfg.mode != 'mil':
        raise ValueError("Only current patient MIL is supported. Historical single/multitask/dual-subtype runners remain in original_codes; D06 uses the registered instance-loss policy.")
    cfg.num_workers = 0
    return _run_mil_experiment(cfg)


def _group_bags(ds, indices):
    """Group dataset indices into per-patient bags.
    Returns list of (patient_id, [idx,...], patient_label)."""
    import collections
    by_pid = collections.OrderedDict()
    for i in indices:
        pid = ds.samples[i][3]
        by_pid.setdefault(pid, []).append(i)
    bags = []
    for pid, idxs in by_pid.items():
        y = int(ds.samples[idxs[0]][2])
        bags.append((pid, idxs, y))
    return bags


def _balanced_bag_order(bags, n_classes, rng):
    """Return a class-balanced ordering of patient bags for one epoch.

    Minority-class bags are oversampled (with replacement) so each class
    contributes ~equally to the epoch's gradient, cutting fold variance on
    imbalanced small-sample tasks. Falls back to a plain shuffle if any class
    is empty or only one class is present.
    """
    import collections
    by_cls = collections.defaultdict(list)
    for b in bags:
        by_cls[b[2]].append(b)
    present = [c for c in range(n_classes) if by_cls[c]]
    if len(present) < 2:
        order = list(bags)
        rng.shuffle(order)
        return order
    target = max(len(by_cls[c]) for c in present)
    out = []
    for c in present:
        pool = list(by_cls[c])
        rng.shuffle(pool)
    # 循环遍历该类别样本池，直到达到 target 个样本。
        reps = (target + len(pool) - 1) // len(pool)
        out.extend((pool * reps)[:target])
    rng.shuffle(out)
    return out


def _encode_bag(ds, idxs, encoder, device, cap, rng=None):
    """Load + encode all image-pairs of a patient bag into z=[N, D]."""
    if len(idxs) > cap:
        idxs = rng.sample(idxs, cap) if rng is not None else idxs[:cap]
    xc = torch.stack([ds[i]["x_central"] for i in idxs], dim=0).to(device)
    xw = torch.stack([ds[i]["x_whorl"] for i in idxs], dim=0).to(device)
    with _autocast(device):
        z, _ = encoder(xc, xw)            # 形状：[N, D]
    return z.float()


def _load_bag_tensors(ds, idxs, device, cap, rng=None):
    if len(idxs) > cap:
        idxs = rng.sample(idxs, cap) if rng is not None else idxs[:cap]
    xc = torch.stack([ds[i]["x_central"] for i in idxs], dim=0).to(device)
    xw = torch.stack([ds[i]["x_whorl"] for i in idxs], dim=0).to(device)
    return xc, xw


def _encode_bag_with_tensors(ds, idxs, encoder, device, cap, rng=None):
    xc, xw = _load_bag_tensors(ds, idxs, device, cap, rng)
    with _autocast(device):
        z, _ = encoder(xc, xw)
    return z.float(), xc, xw


def _image_morph_stats(x):
    """Cheap CCM morphology proxy from grayscale tensors [N,1,H,W].

    These features are deliberately simple and deterministic: intensity density,
    gradient energy, high-gradient fraction, and orientation anisotropy. They act
    as a late-fused morphology token without depending on an external UNet.
    """
    x = x.float()
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    gx_abs = gx.abs()
    gy_abs = gy.abs()
    g_common = torch.sqrt(gx_abs[..., :-1, :] ** 2 + gy_abs[..., :, :-1] ** 2 + 1e-6)
    flat = x.flatten(1)
    gflat = g_common.flatten(1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    bright_frac = (flat > (mean[:, None] + std[:, None])).float().mean(dim=1)
    gmean = gflat.mean(dim=1)
    gstd = gflat.std(dim=1, unbiased=False)
    high_g = (gflat > (gmean[:, None] + gstd[:, None])).float().mean(dim=1)
    gx_m = gx_abs.flatten(1).mean(dim=1)
    gy_m = gy_abs.flatten(1).mean(dim=1)
    anis = (gx_m - gy_m) / (gx_m + gy_m + 1e-6)
    return torch.stack([mean, std, bright_frac, gmean, gstd, high_g, anis], dim=1)


def _bag_morph_features(xc, xw):
    c = _image_morph_stats(xc)
    w = _image_morph_stats(xw)
    pair = torch.cat([c, w, (c - w).abs()], dim=1)  # 形状：[N, 21]
    return pair.mean(dim=0, keepdim=True)


def _encoder_finetune_params(encoder, policy: str):
    if policy == "none":
        return []
    if policy == "all":
        for p in encoder.parameters():
            p.requires_grad_(True)
        return [p for p in encoder.parameters() if p.requires_grad]
    if policy == "high":
        modules = []
        for branch_name in ("branch_c", "branch_w"):
            branch = getattr(encoder, branch_name, None)
            if branch is not None and getattr(branch, "stages", None):
                modules.append(branch.stages[-1])
        for name in ("fuse_c", "fuse_w", "proj"):
            if hasattr(encoder, name):
                modules.append(getattr(encoder, name))
        params = []
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)
                params.append(p)
        return params
    raise ValueError(f"unknown finetune_encoder_policy={policy}")


def _dedup_params(params):
    seen = set()
    out = []
    for p in params:
        ident = id(p)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(p)
    return out


def _stage_threshold_targets(y, device):
    y = y.long()
    return torch.stack([(y >= 1).float(), (y >= 2).float()], dim=1).to(device)


def _stage_threshold_predict(logits):
    return (torch.sigmoid(logits) > 0.5).long().sum(dim=1)


def _stage_threshold_loss(logits, y, pos_weight=None):
    target = _stage_threshold_targets(y, logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _run_mil_experiment(cfg: Stage2Config) -> dict:
    """Per-task Attention-MIL patient-bag training. Each patient is a bag of
    image-pairs; a gated-attention head pools instance embeddings into one
    patient embedding trained directly against the patient label. Evaluation is
    natively patient-level (no post-hoc aggregation needed)."""
    _configure_reproducibility(cfg)
    device = _pick_device(cfg.device)
    run_dir = os.path.join(cfg.output_dir, f"{cfg.run_name}_{cfg.task}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger = TrainLogger(run_dir)
    logger.banner(f"TopoDiSE Stage 2 - RACF MIL patient-bag [{cfg.task}]", {
        "device": str(device), "run_dir": run_dir,
        "parent_ckpt": cfg.parent_ckpt, "task": cfg.task,
        "epochs": cfg.epochs, "lr": cfg.lr, "adapter_rank": cfg.adapter_rank,
        "mil_attn_hidden": cfg.mil_attn_hidden, "mil_dropout": cfg.mil_dropout,
        "mil_bags_per_step": cfg.mil_bags_per_step, "mil_max_bag": cfg.mil_max_bag,
        "mil_pooling": cfg.mil_pooling,
        "pair_mode": cfg.pair_mode, "max_pairs_per_patient": cfg.max_pairs_per_patient,
        "finetune_fusion": cfg.finetune_fusion, "use_tta": cfg.use_tta,
        "use_focal": cfg.use_focal, "use_class_weights": cfg.use_class_weights,
        "use_ordinal": cfg.use_ordinal,
        "use_stage_thresholds": getattr(cfg, "use_stage_thresholds", False),
        "use_morphology_features": getattr(cfg, "use_morphology_features", False),
        "shuffle_morphology_features": getattr(cfg, "shuffle_morphology_features", False),
        "morphology_only": getattr(cfg, "morphology_only", False),
        "use_adapters": getattr(cfg, "use_adapters", True),
        "finetune_encoder_policy": getattr(cfg, "finetune_encoder_policy", "none"),
        "use_feature_diffusion": cfg.use_feature_diffusion,
        "fdiff_gate_mode": cfg.fdiff_gate_mode,
        "fdiff_target_grad_scale": cfg.fdiff_target_grad_scale,
        "smoke": cfg.smoke, "seed": cfg.seed,
        "deterministic": cfg.deterministic,
    })
    logger.event("env", **env_banner())

    train_tf = build_pretrain_transform(image_size=cfg.image_size)
    eval_tf = build_eval_transform(image_size=cfg.image_size)
    full_ds = _make_dataset(cfg, train_tf)
    logger.info(f"dataset: total={len(full_ds)} per_class={full_ds.label_counts}")
    if cfg.few_shot_k > 0:
        full_ds = full_ds.few_shot_subset(cfg.few_shot_k, seed=cfg.seed)
    full_ds_eval = _clone_dataset_with_transform(full_ds, eval_tf)
    _warm_image_cache(full_ds, logger)
    splits = _build_kfold_splits(full_ds, cfg)
    logger.info(
        f"kfold: mode={cfg.kfold_mode} num_folds={cfg.num_folds} "
        f"eval_folds={cfg.eval_folds} evaluated={len(splits)}"
    )

    n_classes = NUM_CLASSES[cfg.task]
    rng = random.Random(cfg.seed)
    t_start = time.time()
    fold_results = _load_partial_fold_results(run_dir, cfg, len(splits), logger)
    completed_folds = {int(r["fold"]) for r in fold_results}

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        if fold in completed_folds:
            logger.info(f"[resume][stage2/mil] fold {fold}/{len(splits)} already complete; skip")
            continue
        train_bags = _group_bags(full_ds, train_idx)
        val_bags = _group_bags(full_ds, val_idx)
        train_bags, val_bags = current_hooks().fold_bags(train_bags, val_bags, cfg, fold)
        logger.banner(f"MIL Fold {fold}/{len(splits)}", {
            "n_train_pairs": len(train_idx), "n_val_pairs": len(val_idx),
            "n_train_bags": len(train_bags), "n_val_bags": len(val_bags),
        })

        encoder, unet, topo_head, pc = _build_from_ckpt(cfg.parent_ckpt, device, cfg)
        apply_freeze_policy(encoder, unet, topo_head, policy="racf_all")
        adapters = attach_adapters(encoder, r=cfg.adapter_rank) if getattr(cfg, "use_adapters", True) else {}
        for adp in adapters.values():
            adp.to(device)
            for p in adp.parameters():
                p.requires_grad_(True)
        encoder_ft_params = _encoder_finetune_params(
            encoder, getattr(cfg, "finetune_encoder_policy", "none"))
        if fold == 1 and encoder_ft_params:
            logger.info(f"  finetune_encoder_policy={cfg.finetune_encoder_policy}: "
                        f"unfroze {sum(p.numel() for p in encoder_ft_params):,} encoder params")

        fusion_params: List[nn.Parameter] = []
        if cfg.finetune_fusion:
            for name in ("fuse_c", "fuse_w", "proj"):
                if hasattr(encoder, name):
                    for p in getattr(encoder, name).parameters():
                        p.requires_grad_(True)
                        fusion_params.append(p)
            if encoder_ft_params:
                fusion_ids = {id(p) for p in fusion_params}
                encoder_ft_params = [p for p in encoder_ft_params if id(p) not in fusion_ids]
            if fold == 1:
                logger.info(f"  finetune_fusion: unfroze {sum(p.numel() for p in fusion_params):,} params")

        morph_only = bool(getattr(cfg, "morphology_only", False))
        use_morph = bool(getattr(cfg, "use_morphology_features", False) or morph_only)
        morph_dim = 21
        morph_proj = None
        head_z_dim = cfg.morph_embed_dim if morph_only else pc["z_dim"]
        if use_morph:
            morph_proj = nn.Sequential(
                nn.LayerNorm(morph_dim),
                nn.Linear(morph_dim, cfg.morph_embed_dim),
                nn.SiLU(),
                nn.Dropout(cfg.classifier_dropout),
            ).to(device)
            if not morph_only:
                head_z_dim += cfg.morph_embed_dim
            if fold == 1:
                logger.info(f"  morphology late-fusion: raw_dim={morph_dim} embed_dim={cfg.morph_embed_dim} "
                            f"shuffle={cfg.shuffle_morphology_features} only={morph_only}")

        if cfg.mil_pooling == "mean":
            pool = _MeanMILPool(pc["z_dim"], dropout=cfg.mil_dropout).to(device)
        else:
            pool = _GatedAttentionPool(pc["z_dim"], hidden=cfg.mil_attn_hidden,
                                       dropout=cfg.mil_dropout).to(device)
        # 有序（CORAL）头仅用于有序任务 pd_stage；
        # 其他任务使用标准无序分类器。
        stage_thresholds = bool(getattr(cfg, "use_stage_thresholds", False)) and cfg.task == "pd_stage"
        ordinal = bool(cfg.use_ordinal) and cfg.task == "pd_stage" and not stage_thresholds
        if stage_thresholds:
            cls_head = ClassifierHead(z_dim=head_z_dim, n_classes=2,
                                      dropout=cfg.classifier_dropout).to(device)
            if fold == 1:
                logger.info("  stage head: pairwise thresholds y>=2 and y>=3")
        elif ordinal:
            cls_head = CoralHead(z_dim=head_z_dim, n_classes=n_classes,
                                 dropout=cfg.classifier_dropout).to(device)
            if fold == 1:
                logger.info(f"  ordinal head: CORAL ({n_classes} classes -> {n_classes-1} cumulative logits)")
        else:
            cls_head = ClassifierHead(z_dim=head_z_dim, n_classes=n_classes,
                                      dropout=cfg.classifier_dropout).to(device)

        # 原型/度量头（借鉴小样本原型分类器），作用于池化后的患者嵌入 m
        # （维度为 head_z_dim）。
        proto_head = None
        use_proto = bool(getattr(cfg, "use_prototype", False))
        if use_proto:
            proto_head = PrototypeHead(
                z_dim=head_z_dim, proj_dim=cfg.proto_dim,
                dropout=cfg.proto_dropout, metric=cfg.proto_metric,
            ).to(device)
            if fold == 1:
                logger.info(
                    f"  prototype head: dim={cfg.proto_dim} dropout={cfg.proto_dropout} "
                    f"metric={cfg.proto_metric} lambda_proto={cfg.lambda_proto} "
                    f"eval_blend={cfg.proto_eval_blend} eval_mode={cfg.proto_eval_mode} "
                    f"support_k={cfg.proto_support_k} fdiff_support_aug={cfg.proto_fdiff_support_aug}")

        trainable: List[nn.Parameter] = []
        for adp in adapters.values():
            trainable += list(adp.parameters())
        trainable += encoder_ft_params
        if morph_proj is not None:
            trainable += list(morph_proj.parameters())
        trainable += list(pool.parameters()) + list(cls_head.parameters()) + fusion_params
        if proto_head is not None:
            trainable += list(proto_head.parameters())

        main_params = (
            [p for adp in adapters.values() for p in adp.parameters()]
            + encoder_ft_params
            + ([] if morph_proj is None else list(morph_proj.parameters()))
            + list(pool.parameters()) + list(cls_head.parameters())
            + ([] if proto_head is None else list(proto_head.parameters()))
        )
        trainable = _dedup_params(trainable)
        main_params = _dedup_params(main_params)
        fdiff_head = None
        fdiff_params: List[nn.Parameter] = []
        schedule = None
        effective_fdiff = bool(cfg.use_feature_diffusion)
        if morph_only:
            effective_fdiff = False
            if fold == 1 and cfg.use_feature_diffusion:
                logger.info("  feature_diffusion: disabled for statistics-only input")
        if cfg.use_feature_diffusion and fold == 1 and (use_morph or stage_thresholds):
            logger.info("  feature_diffusion: enabled with morphology/threshold-aware augmentation path")
        if effective_fdiff:
            fdiff_head = FeatureDiffusionHead(
                z_dim=pc["z_dim"], hidden=cfg.fdiff_hidden,
                layers=cfg.fdiff_layers, t_emb_dim=cfg.fdiff_t_emb_dim,
                n_classes=(n_classes if cfg.fdiff_class_cond else 0),
            ).to(device)
            fdiff_head = current_hooks().feature_head(fdiff_head, cfg)
            fdiff_params = list(fdiff_head.parameters())
            schedule = DiffusionSchedule(T=pc["T"], schedule=pc["schedule"]).to(device)
        param_groups = [{"params": main_params, "lr": cfg.lr}]
        if fdiff_params:
            param_groups.append({"params": fdiff_params,
                                 "lr": cfg.lr * cfg.fdiff_lr_scale})
        if fusion_params:
            param_groups.append({"params": fusion_params,
                                 "lr": cfg.lr * cfg.fusion_lr_scale})
        optim = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        trainable = trainable + fdiff_params
        weight_ema = _WeightEMA(trainable, cfg.weight_ema_decay) if cfg.use_weight_ema else None
        from .runtime.protocol import write as write_metadata
        write_metadata(os.path.join(ckpt_dir, f"fold{fold}_metadata.json"), {
            "trainable_parameters": sum(p.numel() for p in trainable),
            "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
            "encoder_trainable_names": [name for name,p in encoder.named_parameters() if p.requires_grad],
            "optimizer_groups": [{"lr": group["lr"], "parameters": sum(p.numel() for p in group["params"])} for group in optim.param_groups],
            "effective_config": asdict(cfg),
        })

        # 每轮的有效包数（平衡采样会过采样少数类）。
        if getattr(cfg, "mil_balanced_bags", False):
            import collections as _coll
            _cls_counts = _coll.Counter(b[2] for b in train_bags)
            _present = [c for c in _cls_counts if _cls_counts[c] > 0]
            bags_per_epoch = (max(_cls_counts.values()) * len(_present)
                              if len(_present) >= 2 else len(train_bags))
        else:
            bags_per_epoch = len(train_bags)
        bags_per_epoch = current_hooks().sampling_budget(train_bags, cfg, bags_per_epoch)
        steps_per_epoch = max(1, math.ceil(bags_per_epoch / max(1, cfg.mil_bags_per_step)))
        total_steps = max(1, cfg.epochs * steps_per_epoch)
        warmup_steps = max(0, cfg.warmup_epochs * steps_per_epoch)
        if cfg.scheduler == "cosine":
            def _lr_lambda(step_idx: int) -> float:
                if warmup_steps > 0 and step_idx < warmup_steps:
                    return float(step_idx + 1) / float(max(1, warmup_steps))
                remain = max(1, total_steps - warmup_steps)
                tt = min(max(0, step_idx - warmup_steps), remain)
                cos = 0.5 * (1.0 + math.cos(math.pi * tt / remain))
                floor = cfg.min_lr / max(cfg.lr, 1e-12)
                return max(floor, cos)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lr_lambda)
        else:
            lr_scheduler = None

        # 从患者级（包）标签计算频率倒数类别权重。
        counts = [0] * n_classes
        for _, _, y in train_bags:
            if 0 <= y < n_classes:
                counts[y] += 1
        class_counts_dict = {c: counts[c] for c in range(n_classes)}
        if cfg.use_class_weights:
            n_total = max(sum(counts), 1)
            class_weight_t = torch.tensor(
                [n_total / (n_classes * max(counts[c], 1)) for c in range(n_classes)],
                dtype=torch.float32, device=device)
            class_weight_t = _apply_class_weight_multipliers(class_weight_t, cfg)
            logger.info(f"  bag class_weights fold{fold}: {[round(w,3) for w in class_weight_t.tolist()]}")
        else:
            class_weight_t = None

        threshold_pos_weight = None
        if stage_thresholds:
            y_bag = torch.tensor([b[2] for b in train_bags], dtype=torch.long)
            targets = torch.stack([(y_bag >= 1).float(), (y_bag >= 2).float()], dim=1)
            pos = targets.sum(dim=0)
            neg = targets.size(0) - pos
            threshold_pos_weight = (neg / pos.clamp_min(1.0)).to(device)
            if fold == 1:
                logger.info(f"  stage_threshold pos_weight={threshold_pos_weight.tolist()}")

        morph_train_sources, morph_val_sources = current_hooks().morphology_sources(
            train_bags, val_bags, cfg, rng, -1)

        if fold == 1:
            pcount = count_params([("cls_head", cls_head), ("attn_pool", pool)])
            pcount["adapters.trainable"] = sum(p.numel() for adp in adapters.values()
                                               for p in adp.parameters() if p.requires_grad)
            pcount["encoder_ft.trainable"] = sum(p.numel() for p in encoder_ft_params)
            if fdiff_head is not None:
                pcount["fdiff_head"] = sum(p.numel() for p in fdiff_head.parameters())
            if morph_proj is not None:
                pcount["morph_proj"] = sum(p.numel() for p in morph_proj.parameters())
            logger.banner("MIL trainable params", {k: f"{v:,}" for k, v in pcount.items()})
            logger.info(f"trainable total = {sum(p.numel() for p in trainable):,}")

        fold_best_acc = -1.0
        fold_best_epoch = -1
        fold_best_train_acc = -1.0
        fold_best_inst_acc = -1.0
        fold_best_pat_acc = -1.0
        fold_best_subtype_acc = None
        fold_best_subtype_bal_acc = None
        fold_best_per_class_acc = {}
        fold_best_subtype_per_class_acc = {}
        fold_best_patient_predictions = []
        no_improve_epochs = 0

            # 冷启动说明：第一轮需从磁盘解码每个患者包中的图像（缓存尚未预热），
            # 并送入冻结编码器，因此该轮最慢。后续轮次会命中解码图像缓存，
            # 只重新执行编码器前向计算，速度会明显提升（可观察 dt=）。
        logger.info(f"  fold{fold}: 开始训练，首个 epoch 需冷解码+编码 "
                    f"{len(train_bags)} 个患者袋（随后命中缓存提速）")

        for epoch in range(cfg.epochs):
            encoder.train()
            pool.train()
            cls_head.train()
            if proto_head is not None:
                proto_head.train()
            if morph_proj is not None:
                morph_proj.train()
            if getattr(cfg, "mil_balanced_bags", False):
                epoch_bags = _balanced_bag_order(train_bags, n_classes, rng)
            else:
                epoch_bags = list(train_bags)
                rng.shuffle(epoch_bags)
            epoch_bags = current_hooks().bag_order(train_bags, cfg, rng, epoch_bags)
            morph_train_sources, morph_val_sources = current_hooks().morphology_sources(
                train_bags, val_bags, cfg, rng, epoch)
            t_ep = time.time()
            step_logits, step_labels = [], []
            step_fdiff_losses, step_ce_aug_losses = [], []
            step_embs = []
            step_instance_losses = []
            b_correct, b_total = 0, 0
            sum_loss, n_step = 0.0, 0
                # 冷启动的第一轮输出进度，避免首次解码和编码所有包时被误认为卡死。
            _warm_every = max(1, len(epoch_bags) // 4) if epoch == 0 else 0
            for bi, (pid, idxs, y) in enumerate(epoch_bags):
                if morph_only:
                    morph_idxs = morph_train_sources.get(pid, idxs)
                    xc_bag, xw_bag = _load_bag_tensors(full_ds, morph_idxs, device, cfg.mil_max_bag, rng)
                    morph = morph_proj(_bag_morph_features(xc_bag, xw_bag))
                    z = None
                    m = morph
                elif use_morph:
                    z, xc_bag, xw_bag = _encode_bag_with_tensors(full_ds, idxs, encoder, device, cfg.mil_max_bag, rng)
                    morph_idxs = morph_train_sources.get(pid)
                    if morph_idxs is not None:
                        xc_m, xw_m = _load_bag_tensors(full_ds, morph_idxs, device, cfg.mil_max_bag, rng)
                    else:
                        xc_m, xw_m = xc_bag, xw_bag
                    morph = morph_proj(_bag_morph_features(xc_m, xw_m))
                    m, _a = pool(z)                 # 形状：[1, D]
                    m = torch.cat([m, morph], dim=1)
                else:
                    z = _encode_bag(full_ds, idxs, encoder, device, cfg.mil_max_bag, rng)
                    morph = None
                    m, _a = pool(z)                 # 形状：[1, D]
                logit = cls_head(m)             # 形状：[1, C]，有序任务为 [1, C-1]。
                instance_loss = current_hooks().instance_loss(z, morph, cls_head, y, cfg, class_weight_t)
                if instance_loss is not None:
                    step_instance_losses.append(instance_loss)
                step_logits.append(logit)
                step_labels.append(y)
                step_embs.append(m)
                if z is not None and fdiff_head is not None and epoch >= cfg.fdiff_start_epoch:
                    fdiff_head.train()
                    L_fdiff_i, L_ce_aug_i = _feature_aug_step_mil(
                        z, int(y), pool, cls_head, fdiff_head, schedule, cfg,
                        n_classes=n_classes,
                        class_counts=class_counts_dict,
                        class_weight=class_weight_t,
                        ordinal=ordinal,
                        morph_token=(morph if use_morph else None),
                        stage_thresholds=stage_thresholds,
                        threshold_pos_weight=threshold_pos_weight,
                    )
                    step_fdiff_losses.append(L_fdiff_i)
                    step_ce_aug_losses.append(L_ce_aug_i)
                if stage_thresholds:
                    _tp = _stage_threshold_predict(logit)
                else:
                    _tp = coral_predict(logit) if ordinal else logit.argmax(-1)
                b_correct += int(_tp.item() == y)
                b_total += 1
                is_last = (bi == len(epoch_bags) - 1)
                if _warm_every and ((bi + 1) % _warm_every == 0 or is_last):
                    logger.info(f"    [warming] fold{fold} ep1 \u7f16\u7801\u8fdb\u5ea6 "
                                f"{bi + 1}/{len(epoch_bags)} bags "
                                f"({(bi + 1) / max(time.time() - t_ep, 1e-6):.1f} bags/s)")
                if len(step_logits) >= cfg.mil_bags_per_step or is_last:
                    logits = torch.cat(step_logits, dim=0)
                    yt = torch.tensor(step_labels, device=device, dtype=torch.long)
                    if stage_thresholds:
                        L = _stage_threshold_loss(logits, yt, pos_weight=threshold_pos_weight)
                    elif ordinal:
                        L = coral_loss(logits, yt, n_classes, weight=class_weight_t)
                    else:
                        L = _ce_or_focal(logits, yt, cfg, class_weight_t)

                    if step_instance_losses:
                        L = torch.stack(step_instance_losses).mean()

                # 患者嵌入上的辅助分幕原型损失。
                    if proto_head is not None and cfg.lambda_proto > 0 and len(step_embs) >= 2:
                        emb_metric = proto_head(torch.cat(step_embs, dim=0))
                        L_proto = episodic_proto_loss(
                            emb_metric, yt, n_classes, proto_head,
                            temp=cfg.proto_temp, min_per_class=cfg.proto_min_per_class,
                        )
                        L = L + cfg.lambda_proto * L_proto

                    L_fdiff = torch.zeros((), device=device)
                    L_ce_aug = torch.zeros((), device=device)
                    if step_fdiff_losses:
                        L_fdiff = torch.stack(step_fdiff_losses).mean()
                        L_ce_aug = torch.stack(step_ce_aug_losses).mean()
                        L = L + cfg.lambda_fdiff * L_fdiff + cfg.lambda_ce_aug * L_ce_aug
                    optim.zero_grad(set_to_none=True)
                    L.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                    optim.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    if weight_ema is not None:
                        weight_ema.update()
                    sum_loss += L.item()
                    n_step += 1
                    step_logits, step_labels = [], []
                    step_fdiff_losses, step_ce_aug_losses = [], []
                    step_embs = []
                    step_instance_losses = []
            train_acc = b_correct / max(b_total, 1)

            # ---- 评估（患者级+实例级监控） ----
            if weight_ema is not None:
                weight_ema.store_and_copy()
            encoder.eval()
            pool.eval()
            cls_head.eval()
            if proto_head is not None:
                proto_head.eval()
            if fdiff_head is not None:
                fdiff_head.eval()
            if morph_proj is not None:
                morph_proj.eval()
            v_correct, v_total = 0, 0
            inst_correct, inst_total = 0, 0
            per_class_correct, per_class_total = {}, {}
            subtype_correct, subtype_total = 0, 0
            subtype_per_class_correct, subtype_per_class_total = {}, {}
            patient_predictions = []
                # 基于原型的评估：从训练折构建类别原型（仅限无序任务），
                # 再与参数化分类头的预测融合。
            proto_protos = None
            proto_present = None
            support_generated = []
            proto_support_emb = None
            proto_support_labels = None
            proto_eval_on = (proto_head is not None and cfg.proto_eval_blend > 0
                             and not stage_thresholds and not ordinal)
            if proto_eval_on:
                with torch.no_grad():
                    _embs, _labs = [], []
                    for _pid, _idxs, _y in train_bags:
                        _sub = _idxs[:cfg.mil_max_bag]
                        _morph_idxs = morph_train_sources.get(_pid, _idxs)
                        if morph_only:
                            _xc_m, _xw_m = _load_bag_tensors(full_ds_eval, _morph_idxs, device, cfg.mil_max_bag)
                            _morph = morph_proj(_bag_morph_features(_xc_m, _xw_m))
                            _z = None
                            _m = _morph
                        else:
                            _xc = torch.stack([full_ds_eval[i]["x_central"] for i in _sub], dim=0).to(device)
                            _xw = torch.stack([full_ds_eval[i]["x_whorl"] for i in _sub], dim=0).to(device)
                            with _autocast(device):
                                _z, _ = encoder(_xc, _xw)
                            _z = _z.float()
                            _m, _ = pool(_z)
                            if use_morph:
                                if _pid in morph_train_sources:
                                    _xc_m, _xw_m = _load_bag_tensors(full_ds_eval, _morph_idxs, device, cfg.mil_max_bag)
                                else:
                                    _xc_m, _xw_m = _xc, _xw
                                _morph = morph_proj(_bag_morph_features(_xc_m, _xw_m))
                            else:
                                _morph = None
                        if use_morph and not morph_only:
                            _m = torch.cat([_m, _morph], dim=1)
                        _embs.append(proto_head(_m))
                        _labs.append(_y)
                        support_generated.append(False)
                        _aug_by_class = tuple(getattr(cfg, "proto_fdiff_support_aug_by_class", ()) or ())
                        if (_z is not None and fdiff_head is not None and schedule is not None
                                and (cfg.proto_fdiff_support_aug > 0 or _aug_by_class)):
                            _aug_count = int(cfg.proto_fdiff_support_aug)
                            if 0 <= int(_y) < len(_aug_by_class):
                                _aug_count = int(_aug_by_class[int(_y)])
                            _n = _z.size(0)
                            _yv = torch.full((_n,), int(_y), device=device, dtype=torch.long)
                            _T_aug = max(1, min(cfg.fdiff_T_aug, schedule.T - 1))
                            for _ in range(max(0, _aug_count)):
                                _t = torch.randint(1, _T_aug + 1, (_n,), device=device)
                                _eps = torch.randn_like(_z)
                                _zt = q_sample_feature(_z, _t, _eps, schedule.alpha_bar)
                                _zaug_raw = fdiff_head(_zt, _t, _yv if cfg.fdiff_class_cond else None)
                                _zaug = _fdiff_gate(_zaug_raw, _z, cfg)
                                current_hooks().feature_diagnostics(_zaug_raw, _z, _zaug, _yv, cfg, "support")
                                _maug, _ = pool(_zaug)
                                if use_morph:
                                    _maug = torch.cat([_maug, _morph], dim=1)
                                _embs.append(proto_head(_maug))
                                _labs.append(_y)
                                support_generated.append(True)
                    _E = torch.cat(_embs, dim=0)
                    _Lb = torch.tensor(_labs, device=device, dtype=torch.long)
                    proto_support_emb = _E
                    proto_support_labels = _Lb
                    if cfg.proto_eval_mode == "mean":
                        proto_protos, proto_present = PrototypeHead.compute_prototypes(_E, _Lb, n_classes)
            replay_queries = []
            if device.type == "cuda": torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            with torch.no_grad():
                for pid, idxs, y in val_bags:
                    sub = idxs[:cfg.mil_max_bag]
                    morph_idxs = morph_val_sources.get(pid, idxs)
                    if morph_only:
                        xc, xw = _load_bag_tensors(full_ds_eval, morph_idxs, device, cfg.mil_max_bag)
                        morph = morph_proj(_bag_morph_features(xc, xw))
                        z = None
                        m = morph
                    else:
                        xc = torch.stack([full_ds_eval[i]["x_central"] for i in sub], dim=0).to(device)
                        xw = torch.stack([full_ds_eval[i]["x_whorl"] for i in sub], dim=0).to(device)
                        with _autocast(device):
                            z, _ = encoder(xc, xw)
                        z = z.float()
                        m, _a = pool(z)
                        if use_morph:
                            if pid in morph_val_sources:
                                xc_m, xw_m = _load_bag_tensors(full_ds_eval, morph_idxs, device, cfg.mil_max_bag)
                            else:
                                xc_m, xw_m = xc, xw
                            morph = morph_proj(_bag_morph_features(xc_m, xw_m))
                        else:
                            morph = None
                    if use_morph and not morph_only:
                        m = torch.cat([m, morph], dim=1)
                    bag_logit = cls_head(m)
                    if proto_head is not None:
                        replay_queries.append(proto_head(m).detach().cpu())
                    patient_logit = bag_logit
                    if stage_thresholds:
                        pred = int(_stage_threshold_predict(bag_logit).item())
                        inst_pred = _stage_threshold_predict(bag_logit.expand(max(1, len(sub)), -1)) if morph_only else _stage_threshold_predict(cls_head(torch.cat([
                            z, morph_proj(_bag_morph_features(xc, xw)).expand(z.size(0), -1)
                        ], dim=1))) if use_morph else _stage_threshold_predict(cls_head(z))
                    elif ordinal:
                        pred = int(coral_predict(bag_logit).item())
                        if morph_only:
                            inst_pred = coral_predict(bag_logit.expand(max(1, len(sub)), -1))
                        elif use_morph:
                            inst_morph = morph_proj(_bag_morph_features(xc, xw)).expand(z.size(0), -1)
                            inst_pred = coral_predict(cls_head(torch.cat([z, inst_morph], dim=1)))
                        else:
                            inst_pred = coral_predict(cls_head(z))
                    else:
                        pred = int(_binary_threshold_predict(bag_logit, cfg, n_classes).item())
                        if proto_eval_on and proto_support_emb is not None:
                            _q = proto_head(m)
                            if cfg.proto_eval_mode == "mean":
                                _plog = proto_head.distance_logits(_q, proto_protos, temp=cfg.proto_temp)
                                _present = proto_present
                            else:
                                _plog, _present = proto_head.support_logits(
                                    _q, proto_support_emb, proto_support_labels, n_classes,
                                    temp=cfg.proto_temp, mode=cfg.proto_eval_mode,
                                    top_k=cfg.proto_support_k,
                                )
                            _plog = _plog.masked_fill(_present.unsqueeze(0) == 0, -1e4)
                            _blend = float(cfg.proto_eval_blend)
                            _mixed = ((1.0 - _blend) * F.softmax(bag_logit, dim=-1)
                                      + _blend * F.softmax(_plog, dim=-1))
                            patient_logit = torch.log(_mixed.clamp_min(1e-8))
                            pred = int(_binary_threshold_predict(patient_logit, cfg, n_classes).item())
                        if cfg.use_tta:
                            inst_logits = bag_logit.expand(max(1, len(sub)), -1) if morph_only else cls_head(torch.cat([
                                z, morph_proj(_bag_morph_features(xc, xw)).expand(z.size(0), -1)
                            ], dim=1)) if use_morph else _tta_logits(encoder, cls_head, xc, xw, cfg.tta_views)
                        else:
                            inst_logits = bag_logit.expand(max(1, len(sub)), -1) if morph_only else cls_head(torch.cat([
                                z, morph_proj(_bag_morph_features(xc, xw)).expand(z.size(0), -1)
                            ], dim=1)) if use_morph else cls_head(z)
                        inst_pred = _binary_threshold_predict(inst_logits, cfg, n_classes)
                    inst_correct += int((inst_pred == y).sum().item())
                    inst_total += inst_pred.numel()
                    v_correct += int(pred == y)
                    v_total += 1
                    per_class_total[y] = per_class_total.get(y, 0) + 1
                    per_class_correct[y] = per_class_correct.get(y, 0) + int(pred == y)
                    bag_probs = F.softmax(patient_logit.detach().float().cpu(), dim=-1).squeeze(0).tolist()
                    bag_logits = patient_logit.detach().float().cpu().squeeze(0).tolist()
                    cls_probs = F.softmax(bag_logit.detach().float().cpu(), dim=-1).squeeze(0).tolist()
                    cls_logits = bag_logit.detach().float().cpu().squeeze(0).tolist()
                    patient_predictions.append({
                        "patient_id": str(pid),
                        "y_true": int(y),
                        "y_pred": int(pred),
                        "patient_logits": [float(v) for v in bag_logits],
                        "patient_probs": [float(v) for v in bag_probs],
                        "classifier_logits": [float(v) for v in cls_logits],
                        "classifier_probs": [float(v) for v in cls_probs],
                    })
                    if cfg.task == "dm_threeway" and y in (1, 2):
                        subtype_total += 1
                        subtype_correct += int(pred == y)
                        subtype_per_class_total[y] = subtype_per_class_total.get(y, 0) + 1
                        subtype_per_class_correct[y] = subtype_per_class_correct.get(y, 0) + int(pred == y)
            if device.type == "cuda": torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_start
            from .runtime.protocol import write as write_timing
            write_timing(os.path.join(ckpt_dir, f"fold{fold}_epoch{epoch+1}_timing.json"), {
                "patient_inference_seconds": inference_seconds, "n_patients": len(val_bags),
                "scope": "patient validation loop including image load/encode and metrics; support construction excluded",
            })
            val_pat_acc = v_correct / max(v_total, 1)
            inst_acc = inst_correct / max(inst_total, 1)
            per_class_acc = {c: per_class_correct[c] / max(per_class_total[c], 1)
                             for c in per_class_total}
            subtype_acc = subtype_correct / max(subtype_total, 1)
            subtype_per_class_acc = {
                c: subtype_per_class_correct[c] / max(subtype_per_class_total[c], 1)
                for c in subtype_per_class_total
            }
            subtype_bal_acc = (sum(subtype_per_class_acc.values()) / len(subtype_per_class_acc)
                               if subtype_per_class_acc else 0.0)
            if cfg.select_metric == "balanced_patient":
                monitor = (sum(per_class_acc.values()) / len(per_class_acc)) if per_class_acc else 0.0
            elif cfg.select_metric == "subtype_binary":
                monitor = subtype_acc
            elif cfg.select_metric == "balanced_subtype_binary":
                monitor = subtype_bal_acc
            else:
                monitor = val_pat_acc
            ep_dt = time.time() - t_ep
            logger.info(
                f"[mil fold {fold}/{len(splits)} ep {epoch+1:03d}/{cfg.epochs}] "
                f"L={sum_loss/max(n_step,1):.4f} train_acc={train_acc:.3f} "
                f"val_pat_acc={val_pat_acc:.3f} inst_acc={inst_acc:.3f} "
                f"per_class={per_class_acc} subtype_acc={subtype_acc:.3f} "
                f"subtype_bal_acc={subtype_bal_acc:.3f} subtype_per_class={subtype_per_class_acc} "
                f"monitor={monitor:.3f} dt={ep_dt:.1f}s"
            )
            logger.event("epoch_end", stage=2, task=cfg.task, mode="mil", fold=fold,
                         epoch=epoch + 1, train_acc=train_acc, val_acc=val_pat_acc,
                         val_pat_acc=val_pat_acc, inst_acc=inst_acc,
                         per_class_acc=per_class_acc,
                         subtype_acc=subtype_acc,
                         subtype_bal_acc=subtype_bal_acc,
                         subtype_per_class_acc=subtype_per_class_acc,
                         monitor=monitor, select_metric=cfg.select_metric,
                         dt=ep_dt)

            current_hooks().record_epoch(cfg, fold, epoch + 1, patient_predictions, ckpt_dir)
            if monitor > (fold_best_acc + cfg.min_delta):
                fold_best_acc = monitor
                fold_best_epoch = epoch + 1
                fold_best_train_acc = train_acc
                fold_best_inst_acc = inst_acc
                fold_best_pat_acc = val_pat_acc
                fold_best_subtype_acc = subtype_acc if cfg.task == "dm_threeway" else None
                fold_best_subtype_bal_acc = subtype_bal_acc if cfg.task == "dm_threeway" else None
                fold_best_per_class_acc = dict(per_class_acc)
                fold_best_subtype_per_class_acc = dict(subtype_per_class_acc)
                fold_best_patient_predictions = list(patient_predictions)
                no_improve_epochs = 0
                torch.save({
                    "config": asdict(cfg), "fold": fold,
                    "encoder": encoder.state_dict(), "parent_config": pc,
                    "replay": {
                        "queries": torch.cat(replay_queries) if replay_queries else None,
                        "support": proto_support_emb.detach().cpu() if proto_support_emb is not None else None,
                        "support_labels": proto_support_labels.detach().cpu() if proto_support_labels is not None else None,
                        "support_generated": support_generated,
                    },
                    "adapters": {k: v.state_dict() for k, v in adapters.items()},
                    "attn_pool": pool.state_dict(),
                    "morph_proj": (morph_proj.state_dict() if morph_proj is not None else None),
                    "cls_head": cls_head.state_dict(),
                    "proto_head": (proto_head.state_dict() if proto_head is not None else None),
                    "fdiff_head": (fdiff_head.state_dict() if fdiff_head is not None else None),
                    "epoch": epoch + 1, "val_pat_acc": val_pat_acc,
                    "subtype_acc": subtype_acc,
                    "subtype_bal_acc": subtype_bal_acc,
                    "per_class_acc": per_class_acc,
                    "subtype_per_class_acc": subtype_per_class_acc,
                    "patient_predictions": patient_predictions,
                }, os.path.join(ckpt_dir, f"best_fold{fold}.pt"))
            else:
                no_improve_epochs += 1

            if weight_ema is not None:
                weight_ema.restore()

            if cfg.early_stopping and no_improve_epochs >= cfg.patience and (epoch + 1) >= max(3, cfg.warmup_epochs):
                logger.info(f"  -> early stop @ fold {fold} ep {epoch+1} "
                            f"(best={fold_best_acc:.3f} @ ep {fold_best_epoch})")
                break

        fold_results.append({
            "fold": fold, "best_epoch": fold_best_epoch,
            "best_train_acc": fold_best_train_acc,
            "best_val_acc": fold_best_acc,
            "best_image_acc": fold_best_inst_acc,
            "best_patient_acc": fold_best_pat_acc,
            "best_subtype_acc": fold_best_subtype_acc,
            "best_subtype_bal_acc": fold_best_subtype_bal_acc,
            "best_per_class_acc": fold_best_per_class_acc,
            "best_subtype_per_class_acc": fold_best_subtype_per_class_acc,
            "best_patient_predictions": fold_best_patient_predictions,
            "n_train": len(train_bags), "n_val": len(val_bags),
        })
        _save_partial_fold_results(run_dir, cfg, len(splits), fold_results)

    fold_results.sort(key=lambda x: int(x["fold"]))
    best_fold = max(fold_results, key=lambda x: x["best_val_acc"])
    best_ckpt_src = os.path.join(ckpt_dir, f"best_fold{best_fold['fold']}.pt")
    best_ckpt_dst = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_ckpt_src):
        shutil.copyfile(best_ckpt_src, best_ckpt_dst)

    def _mean(vals):
        return sum(vals) / max(len(vals), 1)

    def _std(vals, m):
        return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0

    best_vals = [r["best_val_acc"] for r in fold_results]
    best_trains = [r["best_train_acc"] for r in fold_results]
    best_insts = [r["best_image_acc"] for r in fold_results]
    best_subtypes = [r["best_subtype_acc"] for r in fold_results if r.get("best_subtype_acc") is not None]
    best_subtype_bals = [r["best_subtype_bal_acc"] for r in fold_results if r.get("best_subtype_bal_acc") is not None]
    best_val_mean = _mean(best_vals)
    best_train_mean = _mean(best_trains)
    best_inst_mean = _mean(best_insts)
    best_subtype_mean = _mean(best_subtypes)
    best_subtype_bal_mean = _mean(best_subtype_bals)
    best_val_std = _std(best_vals, best_val_mean)

    logger.banner(f"Stage 2 MIL [{cfg.task}] done", {
        "best_patient_acc(mean)": f"{best_val_mean:.4f}",
        "best_patient_acc(std)": f"{best_val_std:.4f}",
        "best_inst_acc(mean)": f"{best_inst_mean:.4f}",
        "best_subtype_acc(mean)": f"{best_subtype_mean:.4f}" if best_subtypes else "n/a",
        "best_subtype_bal_acc(mean)": f"{best_subtype_bal_mean:.4f}" if best_subtype_bals else "n/a",
        "best_train_acc(mean)": f"{best_train_mean:.4f}",
        "best_gap(mean)": f"{(best_train_mean - best_val_mean):.4f}",
        "evaluated_folds": f"{len(fold_results)}",
        "total_time_s": f"{time.time()-t_start:.1f}",
    })
    logger.close()
    return {
        "task": cfg.task,
        "best_val_acc": best_val_mean,
        "best_val_acc_std": best_val_std,
        "best_image_acc": best_inst_mean,
        "best_patient_acc": best_val_mean,
        "best_subtype_acc": best_subtype_mean if best_subtypes else None,
        "best_subtype_bal_acc": best_subtype_bal_mean if best_subtype_bals else None,
        "select_metric": cfg.select_metric,
        "eval_patient_agg": True,
        "best_train_acc_at_best_epoch": best_train_mean,
        "best_gap": best_train_mean - best_val_mean,
        "evaluated_folds": len(fold_results),
        "num_folds": cfg.num_folds,
        "eval_folds": cfg.eval_folds,
        "n_train": fold_results[0]["n_train"] if fold_results else 0,
        "n_val": fold_results[0]["n_val"] if fold_results else 0,
        "fold_results": fold_results,
    }


# ============================================================================
# 双二分类 MIL 亚型：TD-对-其余 + PIGD-对-其余 -> 融合三分类。
# ============================================================================







# ============================================================================
# 多任务运行器：共享编码器+适配器+3 个分类头。
# ============================================================================
