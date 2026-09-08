"""Stage0：在 CORN-1 上执行 TopoDiSE 自监督预训练。

损失：L = lambda_simple * ||eps - eps_hat||^2 + lambda_topo * BCE(psi(z), targets)。
烟雾测试模式的拓扑目标为随机二值图，用作 Frangi 缓存的占位数据。
"""
import os
import sys
import json
import time
import argparse
from dataclasses import asdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ALT_CODES_ROOT = os.path.join(PROJECT_ROOT, "codes")
for p in (PROJECT_ROOT, ALT_CODES_ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from core.progress import ETAEstimator, format_duration
from topodise import (
    TopoDiSEEncoder, EncoderConfig,
    ConditionalUNet, UNetConfig,
    TopoHead, DiffusionSchedule, q_sample,
    ModelEMA,
    apply_freeze_policy,
    compute_topo_target, nt_xent_loss,
    TrainLogger, EMAMeter, RateMeter, count_params, grad_global_norm, env_banner,
)
from topodise.data import CORN1Dataset, build_pretrain_transform
from topodise.reproducibility import (
    capture_rng_state,
    configure_reproducibility,
    restore_rng_state,
)
from .config import Stage0Config


def _pick_device(dev: str) -> torch.device:
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_input_path(path: str) -> str:
    """相对 codes 根目录解析默认路径，不依赖启动时的工作目录。"""
    if os.path.isabs(path):
        return os.path.normpath(path)
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _max_epoch_from_events(events_path: str) -> int:
    if not os.path.exists(events_path):
        return 0
    max_ep = 0
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") == "epoch_end" and int(rec.get("stage", -1)) == 0:
                    ep = int(rec.get("epoch", 0))
                    if ep > max_ep:
                        max_ep = ep
    except Exception:
        return 0
    return max_ep


def _torch_save_atomic(payload: dict, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_single_experiment(cfg: Stage0Config) -> dict:
    rng_strategy = configure_reproducibility(cfg.reproducibility)
    device = _pick_device(cfg.device)

    cfg.corn1_root = _resolve_input_path(cfg.corn1_root)
    cfg.output_dir = _resolve_output_path(cfg.output_dir)

    run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger = TrainLogger(run_dir)
    logger.banner("TopoDiSE Stage 0 - SSL pretraining on CORN-1", {
        "device": str(device), "run_dir": run_dir,
        "corn1_root": cfg.corn1_root, "image_size": cfg.image_size,
        "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay,
        "T": cfg.T, "schedule": cfg.schedule,
        "z_dim": cfg.z_dim, "z_dropout": cfg.z_dropout,
        "lambda_simple": cfg.lambda_simple, "lambda_topo": cfg.lambda_topo,
        "resume": cfg.resume, "strict_resume": getattr(cfg, "strict_resume", True),
        "smoke": cfg.smoke,
        "rng_strategy": rng_strategy["strategy"],
        "reproducibility": cfg.reproducibility,
        "fixed_torch_seed": rng_strategy.get("seed"),
        "reference_recipe": rng_strategy.get("recipe"),
    })
    logger.event("env", **env_banner())
    logger.event("rng_strategy", **rng_strategy)

    transform = build_pretrain_transform(image_size=cfg.image_size)
    transform2 = build_pretrain_transform(image_size=cfg.image_size)
    dataset = CORN1Dataset(
        cfg.corn1_root, transform=transform, transform_paired=transform2,
        load_masks=(cfg.topo_target == "gt"), image_size=cfg.image_size,
    )
    if cfg.expected_samples is not None and not cfg.smoke and len(dataset) != cfg.expected_samples:
        raise RuntimeError(
            "CORN-1 does not match the c8bf6b548656 reference dataset: "
            f"expected {cfg.expected_samples} images, found {len(dataset)}. "
            "Set expected_samples=None only when intentionally training a different recipe."
        )
    loader_workers = cfg.num_workers if not cfg.smoke else 0
    pin_memory = device.type == "cuda" if cfg.pin_memory is None else bool(cfg.pin_memory)
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": loader_workers,
        "drop_last": True,
        "pin_memory": bool(pin_memory and device.type == "cuda"),
        "persistent_workers": bool(cfg.persistent_workers and loader_workers > 0),
    }
    if loader_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
    loader = DataLoader(dataset, **loader_kwargs)
    logger.info(f"dataset: size={len(dataset)} batches/epoch={len(loader)}")

    enc_cfg = EncoderConfig(
        in_channels=1,
        base_channels=cfg.encoder_base_channels,
        channel_mult=tuple(cfg.encoder_channel_mult),
        blocks_per_stage=tuple(cfg.encoder_blocks_per_stage),
        z_dim=cfg.z_dim,
    )
    encoder = TopoDiSEEncoder(enc_cfg).to(device)
    unet_cfg = UNetConfig(
        in_channels=1, out_channels=1,
        model_channels=cfg.unet_model_channels,
        channel_mult=tuple(cfg.unet_channel_mult),
        z_dim=cfg.z_dim, image_size=cfg.image_size,
    )
    unet = ConditionalUNet(unet_cfg).to(device)
    topo_head = TopoHead(z_dim=cfg.z_dim).to(device)
    if cfg.channels_last and device.type == "cuda":
        encoder = encoder.to(memory_format=torch.channels_last)
        unet = unet.to(memory_format=torch.channels_last)
    apply_freeze_policy(encoder, unet, topo_head, policy="none")

    pcount = count_params([("encoder", encoder), ("unet", unet), ("topo_head", topo_head)])
    logger.banner("Model parameters", {k: f"{v:,}" for k, v in pcount.items()})
    logger.event("param_count", **pcount)

    schedule = DiffusionSchedule(T=cfg.T, schedule=cfg.schedule).to(device)
    params = list(encoder.parameters()) + list(unet.parameters()) + list(topo_head.parameters())
    optim_kwargs = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if cfg.fused_adamw and device.type == "cuda":
        try:
            optim = torch.optim.AdamW(params, fused=True, **optim_kwargs)
        except (TypeError, RuntimeError):
            optim = torch.optim.AdamW(params, **optim_kwargs)
    else:
        optim = torch.optim.AdamW(params, **optim_kwargs)
    ema = ModelEMA(nn.ModuleDict(dict(encoder=encoder, unet=unet, topo_head=topo_head)),
                   decay=cfg.ema_decay)

    ema_loss = EMAMeter(decay=0.98)
    rate = RateMeter(window=20)
    log_every = max(1, len(loader) // 5) if not cfg.smoke else max(1, len(loader) // 2)

    best_loss = float("inf")
    start_epoch = 0
    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "ema_best.pt")
    events_path = os.path.join(run_dir, "events.jsonl")
    resume_path = cfg.resume_ckpt.strip() if getattr(cfg, "resume_ckpt", "") else ""
    if cfg.resume:
        if resume_path and os.path.exists(resume_path):
            load_path = resume_path
        elif os.path.exists(last_ckpt_path):
            load_path = last_ckpt_path
        elif os.path.exists(best_ckpt_path):
            load_path = best_ckpt_path
        else:
            load_path = ""

        if load_path:
            state = torch.load(load_path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(state["encoder"])
            unet.load_state_dict(state["unet"])
            topo_head.load_state_dict(state["topo_head"])
            if "optim" in state:
                optim.load_state_dict(state["optim"])
            if "ema" in state:
                ema.load_state_dict(state["ema"])
            best_loss = float(state.get("best_loss", best_loss))
            start_epoch = int(state.get("epoch", 0))
            if not torch.isfinite(torch.tensor(best_loss)):
                if os.path.exists(best_ckpt_path):
                    try:
                        best_state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
                        best_loss = float(best_state.get("best_loss", best_loss))
                    except Exception:
                        pass

            max_logged_epoch = _max_epoch_from_events(events_path)
            if getattr(cfg, "strict_resume", True) and max_logged_epoch > start_epoch:
                raise RuntimeError(
                    "stale resume checkpoint detected: "
                    f"loaded_epoch={start_epoch}, max_logged_epoch={max_logged_epoch}, ckpt={load_path}. "
                    "This usually happens when ema_best.pt was renamed as last.pt. "
                    "Please restore the real last.pt (latest state with optimizer), "
                    "or set strict_resume=False to force resume from this stale checkpoint."
                )
            logger.info(f"resume enabled: loaded {load_path} @ epoch={start_epoch}, best_loss={best_loss:.4f}")
            if state.get("rng_state"):
                restore_rng_state(state["rng_state"])
                logger.info("resume RNG state restored")
            else:
                logger.info("[warn] legacy checkpoint has no RNG state; resumed trajectory is not exact")

    if start_epoch >= cfg.epochs:
        logger.info(f"already finished: start_epoch={start_epoch} >= epochs={cfg.epochs}, skip training")
        logger.close()
        return {
            "best_loss": best_loss,
            "ckpt_path": os.path.join(ckpt_dir, "ema_best.pt"),
            "start_epoch": start_epoch,
            "end_epoch": start_epoch,
        }

    t_start = time.time()
    epoch_progress = ETAEstimator(cfg.epochs, initial_completed=start_epoch, window=5)
    step_progress = ETAEstimator(
        cfg.epochs * len(loader),
        initial_completed=start_epoch * len(loader),
        window=20,
    )
    for epoch in range(start_epoch, cfg.epochs):
        encoder.train()
        unet.train()
        topo_head.train()
        n, s_simple, s_topo, s_gn = 0, 0.0, 0.0, 0.0
        s_contrast = 0.0
        t_ep = time.time()
        for it, batch in enumerate(loader):
            t_b = time.time()
            non_blocking = bool(cfg.non_blocking_transfers and pin_memory and device.type == "cuda")
            x_full = batch["x_full"].to(device, non_blocking=non_blocking)
            xc = batch["x_central"].to(device, non_blocking=non_blocking)
            xw = batch["x_whorl"].to(device, non_blocking=non_blocking)
            if cfg.channels_last and device.type == "cuda":
                x_full = x_full.contiguous(memory_format=torch.channels_last)
                xc = xc.contiguous(memory_format=torch.channels_last)
                xw = xw.contiguous(memory_format=torch.channels_last)
            B = x_full.size(0)

            def amp_context():
                return (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if cfg.amp and device.type == "cuda" else nullcontext()
                )

            with amp_context():
                z, _ = encoder(xc, xw)
                if cfg.z_dropout > 0:
                    drop_mask = (torch.rand(B, 1, device=device) < cfg.z_dropout).float()
                    z_drop = z * (1 - drop_mask)
                else:
                    z_drop = z

                t = torch.randint(0, schedule.T, (B,), device=device)
                noise = torch.randn_like(x_full)
                x_t = q_sample(x_full, t, noise, schedule.alpha_bar)
                eps_hat = unet(x_t, t, z_drop)
                L_simple = F.mse_loss(eps_hat.float(), noise.float())

            with torch.no_grad():
                topo_mask = batch.get("x_mask")
                topo_mask = topo_mask.to(device, non_blocking=non_blocking) if topo_mask is not None else None
                target = compute_topo_target(
                    x_full, mode=cfg.topo_target, out_size=32,
                    sigmas=tuple(cfg.frangi_sigmas),
                    beta=cfg.frangi_beta, c=cfg.frangi_c,
                    mask=topo_mask,
                )
            with amp_context():
                psi_logits = topo_head(z)
            L_topo = F.binary_cross_entropy_with_logits(psi_logits.float(), target.float())

            L = cfg.lambda_simple * L_simple + cfg.lambda_topo * L_topo
            if cfg.use_view_contrastive:
                with amp_context():
                    _z2, (cls_c, cls_w), _ = encoder(xc, xw, return_cls=True)
                L_contrast = nt_xent_loss(cls_c, cls_w, tau=cfg.tau_contrastive)
                L = L + cfg.lambda_contrastive * L_contrast
            else:
                L_contrast = None
            optim.zero_grad(set_to_none=True)
            L.backward()
            gn = grad_global_norm(params)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()
            ema.update(nn.ModuleDict(dict(encoder=encoder, unet=unet, topo_head=topo_head)))

            n += B
            s_simple += L_simple.item() * B
            s_topo += L_topo.item() * B
            s_gn += gn * B
            if L_contrast is not None:
                s_contrast += L_contrast.item() * B
            rate.update(time.time() - t_b, B)
            ema_loss.update(L.item())
            step_state = step_progress.update(
                epoch * len(loader) + it + 1,
                interval_seconds=time.time() - t_b,
            )

            if (it + 1) % log_every == 0 or (it + 1) == len(loader):
                _cstr = f" L_contrast={L_contrast.item():.4f}" if L_contrast is not None else ""
                logger.info(
                    f"  [ep {epoch+1:03d}/{cfg.epochs} it {it+1:04d}/{len(loader)}] "
                    f"L={L.item():.4f} (ema={ema_loss.value:.4f}) "
                    f"L_simple={L_simple.item():.4f} L_topo={L_topo.item():.4f}{_cstr} "
                    f"|g|={gn:.3f} sps={rate.samples_per_sec():.1f}"
                )
                logger.info(
                    f"  [progress] stage=topodise_stage0 "
                    f"steps={step_state['completed_units']}/{step_state['total_units']} "
                    f"({step_state['progress_pct']:.1f}%) "
                    f"elapsed={format_duration(step_state['elapsed_seconds'])} "
                    f"ETA={format_duration(step_state['eta_seconds'])} "
                    f"finish={step_state['estimated_finish_at']} "
                    f"quality={step_state['eta_quality']}"
                )
                logger.event("step_progress", stage="topodise_stage0", **step_state)

        ep_simple = s_simple / max(n, 1)
        ep_topo = s_topo / max(n, 1)
        ep_contrast = s_contrast / max(n, 1)
        ep_total = cfg.lambda_simple * ep_simple + cfg.lambda_topo * ep_topo
        if cfg.use_view_contrastive:
            ep_total = ep_total + cfg.lambda_contrastive * ep_contrast
        ep_dt = time.time() - t_ep
        epoch_state = epoch_progress.update(epoch + 1, interval_seconds=ep_dt)
        logger.info(
            f"[ep {epoch+1:03d}/{cfg.epochs}] L={ep_total:.4f} "
            f"L_simple={ep_simple:.4f} L_topo={ep_topo:.4f} L_contrast={ep_contrast:.4f} "
            f"|g|_avg={s_gn/max(n,1):.3f} dt={ep_dt:.1f}s sps={rate.samples_per_sec():.1f} "
            f"elapsed={logger.elapsed():.0f}s"
        )
        logger.info(
            f"[ETA] stage=topodise_stage0 epoch={epoch+1}/{cfg.epochs} "
            f"({epoch_state['progress_pct']:.1f}%) epoch_s={ep_dt:.1f} "
            f"rolling_epoch_s={epoch_state['rolling_unit_seconds']} "
            f"elapsed={format_duration(epoch_state['elapsed_seconds'])} "
            f"ETA={format_duration(epoch_state['eta_seconds'])} "
            f"finish={epoch_state['estimated_finish_at']} "
            f"quality={epoch_state['eta_quality']}"
        )
        logger.event("epoch_end", stage=0, epoch=epoch + 1,
                     L=ep_total, L_simple=ep_simple, L_topo=ep_topo,
                     L_contrast=ep_contrast,
                     grad_norm_avg=s_gn / max(n, 1), dt=ep_dt,
                     sps=rate.samples_per_sec(), **epoch_state)

        if ep_total < best_loss:
            best_loss = ep_total
            ckpt = {
                "config": asdict(cfg),
                "encoder": encoder.state_dict(),
                "unet": unet.state_dict(),
                "topo_head": topo_head.state_dict(),
                "ema": ema.state_dict(),
                "best_loss": best_loss,
                "rng_state": capture_rng_state(),
                "rng_strategy": rng_strategy,
                "epoch": epoch + 1,
            }
            _torch_save_atomic(ckpt, os.path.join(ckpt_dir, "ema_best.pt"))
            logger.info(f"  -> saved best ckpt @ ep {epoch+1} (L={best_loss:.4f})")

        # 始终保留可续训检查点，以便在外部中断后恢复。
        _torch_save_atomic({
            "config": asdict(cfg),
            "encoder": encoder.state_dict(),
            "unet": unet.state_dict(),
            "topo_head": topo_head.state_dict(),
            "ema": ema.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "rng_state": capture_rng_state(),
            "rng_strategy": rng_strategy,
        }, last_ckpt_path)

    logger.banner("Stage 0 done", {
        "best_loss": f"{best_loss:.4f}",
        "ckpt": os.path.join(ckpt_dir, "ema_best.pt"),
        "last_ckpt": last_ckpt_path,
        "total_time_s": f"{time.time()-t_start:.1f}",
    })
    logger.close()
    return {
        "best_loss": best_loss,
        "ckpt_path": os.path.join(ckpt_dir, "ema_best.pt"),
        "last_ckpt_path": last_ckpt_path,
        "start_epoch": start_epoch,
        "end_epoch": cfg.epochs,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corn1-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Stage0Config()
    if args.smoke:
        cfg.epochs = 1
        cfg.batch_size = 2
        cfg.image_size = 64
        cfg.encoder_base_channels = 16
        cfg.encoder_channel_mult = (1, 2, 2, 2)
        cfg.encoder_blocks_per_stage = (1, 1, 1, 1)
        cfg.unet_model_channels = 16
        cfg.unet_channel_mult = (1, 2, 2)
        cfg.z_dim = 64
        cfg.T = 50
        cfg.num_workers = 0
        cfg.run_name = "smoke_stage0"
    for k in ("corn1_root", "epochs", "batch_size", "image_size", "run_name", "output_dir", "device"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    out = run_single_experiment(cfg)
    print(json.dumps(out, indent=2))
