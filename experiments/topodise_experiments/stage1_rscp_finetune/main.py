"""Stage1（RSCP）：区域专用的持续预训练。"""
import os
import sys
import json
import time
import argparse
from dataclasses import asdict
from contextlib import nullcontext
import hashlib

import torch
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
    ModelEMA, apply_freeze_policy, region_infonce_loss,
    compute_topo_target,
    TrainLogger, EMAMeter, RateMeter, count_params, grad_global_norm, env_banner,
)
from topodise.data import RegionPairDataset, build_pretrain_transform
from topodise.reproducibility import (
    capture_rng_state,
    configure_reproducibility,
    restore_rng_state,
)
from .config import Stage1Config


def _pick_device(dev):
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_input_path(path: str) -> str:
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
                if rec.get("kind") == "epoch_end" and int(rec.get("stage", -1)) == 1:
                    ep = int(rec.get("epoch", 0))
                    if ep > max_ep:
                        max_ep = ep
    except Exception:
        return 0
    return max_ep


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_save_atomic(payload: dict, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _build_from_ckpt(ckpt_path, device, encoder_weights="raw"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg0 = ckpt.get("parent_config") or ckpt["config"]
    enc_cfg = EncoderConfig(
        in_channels=1,
        base_channels=cfg0["encoder_base_channels"],
        channel_mult=tuple(cfg0["encoder_channel_mult"]),
        blocks_per_stage=tuple(cfg0["encoder_blocks_per_stage"]),
        z_dim=cfg0["z_dim"],
    )
    unet_cfg = UNetConfig(
        in_channels=1, out_channels=1,
        model_channels=cfg0["unet_model_channels"],
        channel_mult=tuple(cfg0["unet_channel_mult"]),
        z_dim=cfg0["z_dim"], image_size=cfg0["image_size"],
    )
    encoder = TopoDiSEEncoder(enc_cfg).to(device)
    unet = ConditionalUNet(unet_cfg).to(device)
    topo_head = TopoHead(z_dim=cfg0["z_dim"]).to(device)
    if encoder_weights == "raw":
        encoder_state = ckpt["encoder"]
    elif encoder_weights == "ema":
        if "ema_encoder" in ckpt:
            encoder_state = ckpt["ema_encoder"]
        elif "ema" in ckpt:
            prefix = "encoder."
            encoder_state = {
                key[len(prefix):]: value
                for key, value in ckpt["ema"].items()
                if key.startswith(prefix)
            }
            if not encoder_state:
                raise KeyError("checkpoint EMA state does not contain encoder.* parameters")
        else:
            raise KeyError("checkpoint does not contain EMA encoder weights")
    else:
        raise ValueError("parent_encoder_weights must be 'raw' or 'ema'")
    encoder.load_state_dict(encoder_state, strict=True)
    unet.load_state_dict(ckpt["unet"])
    topo_head.load_state_dict(ckpt["topo_head"])
    return encoder, unet, topo_head, cfg0


def run_single_experiment(cfg: Stage1Config) -> dict:
    rng_strategy = configure_reproducibility(cfg.reproducibility)
    device = _pick_device(cfg.device)

    cfg.parent_ckpt = _resolve_input_path(cfg.parent_ckpt)
    cfg.pd_json = _resolve_input_path(cfg.pd_json)
    cfg.base_path = _resolve_input_path(cfg.base_path)
    if cfg.region_pair_manifest:
        cfg.region_pair_manifest = _resolve_input_path(cfg.region_pair_manifest)
    cfg.output_dir = _resolve_output_path(cfg.output_dir)
    run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger = TrainLogger(run_dir)
    logger.banner("TopoDiSE Stage 1 - RSCP region-specialized continual pretraining", {
        "device": str(device), "run_dir": run_dir,
        "parent_ckpt": cfg.parent_ckpt,
        "pd_json": cfg.pd_json, "base_path": cfg.base_path,
        "patient_groups": cfg.patient_groups,
        "allow_dm_disease": cfg.allow_dm_disease,
        "parent_encoder_weights": cfg.parent_encoder_weights,
        "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay,
        "lambda_simple": cfg.lambda_simple,
        "lambda_topo": cfg.lambda_topo, "lambda_region": cfg.lambda_region,
        "tau_region": cfg.tau_region,
        "smoke": cfg.smoke,
        "rng_strategy": rng_strategy["strategy"],
        "reproducibility": cfg.reproducibility,
        "fixed_torch_seed": rng_strategy.get("seed"),
        "reference_recipe": rng_strategy.get("recipe"),
    })
    logger.event("env", **env_banner())
    logger.event("rng_strategy", **rng_strategy)

    encoder, unet, topo_head, parent_cfg = _build_from_ckpt(
        cfg.parent_ckpt, device, encoder_weights=cfg.parent_encoder_weights,
    )
    parent_checkpoint_sha256 = _sha256_file(cfg.parent_ckpt)
    if cfg.channels_last and device.type == "cuda":
        encoder = encoder.to(memory_format=torch.channels_last)
        unet = unet.to(memory_format=torch.channels_last)
    freeze_info = apply_freeze_policy(encoder, unet, topo_head, policy="rscp_high")
    logger.banner("Freeze policy (rscp_high)",
                  {k: str(v) for k, v in freeze_info.items()})
    schedule = DiffusionSchedule(T=parent_cfg["T"], schedule=parent_cfg["schedule"]).to(device)

    transform = build_pretrain_transform(image_size=parent_cfg["image_size"])
    strict = not cfg.smoke
    if cfg.region_pair_manifest:
        ds = RegionPairDataset(manifest_path=cfg.region_pair_manifest, base_path=cfg.base_path,
                               transform=transform, strict_whitelist=strict,
                               allow_dm_disease=cfg.allow_dm_disease)
    else:
        with open(cfg.pd_json, "r") as f:
            patients = json.load(f)
        patients = [p for p in patients if p.get("group") in tuple(cfg.patient_groups)]
        ds = RegionPairDataset(patients=patients, base_path=cfg.base_path, transform=transform,
                               strict_whitelist=strict,
                               allow_dm_disease=cfg.allow_dm_disease)
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
    loader = DataLoader(ds, **loader_kwargs)
    logger.info(f"region pairs: {len(ds)} batches/epoch={len(loader)}")
    reference_pd_recipe = (
        not cfg.smoke
        and cfg.region_pair_manifest is None
        and tuple(cfg.patient_groups) == ("control", "control_cataract")
        and not cfg.allow_dm_disease
    )
    if (reference_pd_recipe and cfg.expected_region_pairs is not None
            and len(ds) != cfg.expected_region_pairs):
        raise RuntimeError(
            "Stage1 data does not match the c8bf6b548656 reference dataset: "
            f"expected {cfg.expected_region_pairs} region pairs, found {len(ds)}. "
            "Set expected_region_pairs=None only when intentionally training a different recipe."
        )
    if getattr(ds, "skipped_duplicate_filenames", 0):
        logger.info(
            "skipped unlabeled pairs with filenames duplicated in pd_dataset_en/dm_dataset_en: "
            f"{ds.skipped_duplicate_filenames}"
        )

    trainable_params = [p for p in encoder.parameters() if p.requires_grad]
    pcount = count_params([("encoder", encoder), ("unet(frozen)", unet),
                            ("topo_head(frozen)", topo_head)])
    logger.banner("Model parameters", {k: f"{v:,}" for k, v in pcount.items()})
    logger.event("param_count", **pcount)
    optim_kwargs = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if cfg.fused_adamw and device.type == "cuda":
        try:
            optim = torch.optim.AdamW(trainable_params, fused=True, **optim_kwargs)
        except (TypeError, RuntimeError):
            optim = torch.optim.AdamW(trainable_params, **optim_kwargs)
    else:
        optim = torch.optim.AdamW(trainable_params, **optim_kwargs)
    ema = ModelEMA(encoder, decay=cfg.ema_decay)

    ema_loss = EMAMeter(decay=0.98)
    rate = RateMeter(window=20)
    log_every = max(1, len(loader) // 5) if not cfg.smoke else max(1, len(loader) // 2)

    best = float("inf")
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
            encoder.load_state_dict(state["encoder"], strict=False)
            unet.load_state_dict(state["unet"])
            topo_head.load_state_dict(state["topo_head"])
            if "optim" in state:
                optim.load_state_dict(state["optim"])
            if "ema" in state:
                ema.load_state_dict(state["ema"])
            elif "ema_encoder" in state:
                ema.load_state_dict(state["ema_encoder"])
            best = float(state.get("best_loss", best))
            start_epoch = int(state.get("epoch", 0))
            max_logged_epoch = _max_epoch_from_events(events_path)
            if getattr(cfg, "strict_resume", True) and max_logged_epoch > start_epoch:
                raise RuntimeError(
                    "stale Stage1 resume checkpoint detected: "
                    f"loaded_epoch={start_epoch}, max_logged_epoch={max_logged_epoch}, ckpt={load_path}. "
                    "Please restore the real last.pt or set strict_resume=False."
                )
            logger.info(f"resume enabled: loaded {load_path} @ epoch={start_epoch}, best_loss={best:.4f}")
            if state.get("rng_state"):
                restore_rng_state(state["rng_state"])
                logger.info("resume RNG state restored")
            else:
                logger.info("[warn] legacy checkpoint has no RNG state; resumed trajectory is not exact")

    if start_epoch >= cfg.epochs:
        logger.info(f"already finished: start_epoch={start_epoch} >= epochs={cfg.epochs}, skip training")
        logger.close()
        return {
            "best_loss": best,
            "ckpt_path": best_ckpt_path,
            "last_ckpt_path": last_ckpt_path,
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
        n, s_simple, s_topo, s_region, s_gn = 0, 0.0, 0.0, 0.0, 0.0
        t_ep = time.time()
        for it, batch in enumerate(loader):
            t_b = time.time()
            non_blocking = bool(cfg.non_blocking_transfers and pin_memory and device.type == "cuda")
            xc = batch["x_central"].to(device, non_blocking=non_blocking)
            xw = batch["x_whorl"].to(device, non_blocking=non_blocking)
            if cfg.channels_last and device.type == "cuda":
                xc = xc.contiguous(memory_format=torch.channels_last)
                xw = xw.contiguous(memory_format=torch.channels_last)
            B = xc.size(0)
            def amp_context():
                return (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if cfg.amp and device.type == "cuda" else nullcontext()
                )

            with amp_context():
                z, (cls_c, cls_w), _feats = encoder(xc, xw, return_cls=True)

        # 扩散去噪器默认只记录日志；可选地将生成信号反传到编码器。
            t = torch.randint(0, schedule.T, (B,), device=device)
            noise = torch.randn_like(xc)
            if cfg.train_denoiser:
                with amp_context():
                    eps_hat = unet(x_t := q_sample(xc, t, noise, schedule.alpha_bar), t, z)
                L_simple = F.mse_loss(eps_hat.float(), noise.float())
            else:
                with torch.no_grad():
                    x_t = q_sample(xc, t, noise, schedule.alpha_bar)
                    with amp_context():
                        eps_hat = unet(x_t, t, z.detach())
                    L_simple = F.mse_loss(eps_hat.float(), noise.float())
            with torch.no_grad():
                target = compute_topo_target(
                    xc, mode=cfg.topo_target, out_size=32,
                    sigmas=tuple(cfg.frangi_sigmas),
                    beta=cfg.frangi_beta, c=cfg.frangi_c,
                )
            with amp_context():
                psi_logits = topo_head(z)
            L_topo = F.binary_cross_entropy_with_logits(psi_logits.float(), target.float())
            pair_mask = batch.get("is_paired")
            if pair_mask is None:
                pair_mask = torch.ones(B, dtype=torch.bool, device=device)
            else:
                pair_mask = pair_mask.to(device=device, dtype=torch.bool)
            if bool(pair_mask.all()):
                # 保持 c8bf6b548656 全配对路径与历史直接调用一致，
                # 不执行布尔索引复制。
                L_region = region_infonce_loss(cls_c, cls_w, tau=cfg.tau_region)
            else:
                L_region = region_infonce_loss(
                    cls_c[pair_mask], cls_w[pair_mask], tau=cfg.tau_region,
                )

            L = cfg.lambda_region * L_region + cfg.lambda_topo * L_topo
            if cfg.train_denoiser:
                L = L + cfg.lambda_simple * L_simple
            optim.zero_grad(set_to_none=True)
            L.backward()
            gn = grad_global_norm(trainable_params)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optim.step()
            ema.update(encoder)

            n += B
            s_simple += L_simple.item() * B
            s_topo += L_topo.item() * B
            s_region += L_region.item() * B
            s_gn += gn * B
            rate.update(time.time() - t_b, B)
            ema_loss.update(L.item())
            step_state = step_progress.update(
                epoch * len(loader) + it + 1,
                interval_seconds=time.time() - t_b,
            )

            if (it + 1) % log_every == 0 or (it + 1) == len(loader):
                logger.info(
                    f"  [ep {epoch+1:03d}/{cfg.epochs} it {it+1:04d}/{len(loader)}] "
                    f"L={L.item():.4f} (ema={ema_loss.value:.4f}) "
                    f"L_region={L_region.item():.4f} L_topo={L_topo.item():.4f} "
                    f"L_simple(frozen)={L_simple.item():.4f} "
                    f"|g|={gn:.3f} sps={rate.samples_per_sec():.1f}"
                )
                logger.info(
                    f"  [progress] stage=topodise_stage1 "
                    f"steps={step_state['completed_units']}/{step_state['total_units']} "
                    f"({step_state['progress_pct']:.1f}%) "
                    f"elapsed={format_duration(step_state['elapsed_seconds'])} "
                    f"ETA={format_duration(step_state['eta_seconds'])} "
                    f"finish={step_state['estimated_finish_at']} "
                    f"quality={step_state['eta_quality']}"
                )
                logger.event("step_progress", stage="topodise_stage1", **step_state)

        ep_simple = s_simple / max(n, 1)
        ep_topo = s_topo / max(n, 1)
        ep_region = s_region / max(n, 1)
        tot = cfg.lambda_topo * ep_topo + cfg.lambda_region * ep_region
        ep_dt = time.time() - t_ep
        epoch_state = epoch_progress.update(epoch + 1, interval_seconds=ep_dt)
        logger.info(
            f"[ep {epoch+1:03d}/{cfg.epochs}] L={tot:.4f} L_region={ep_region:.4f} "
            f"L_topo={ep_topo:.4f} L_simple(frozen)={ep_simple:.4f} "
            f"|g|_avg={s_gn/max(n,1):.3f} dt={ep_dt:.1f}s elapsed={logger.elapsed():.0f}s"
        )
        logger.info(
            f"[ETA] stage=topodise_stage1 epoch={epoch+1}/{cfg.epochs} "
            f"({epoch_state['progress_pct']:.1f}%) epoch_s={ep_dt:.1f} "
            f"rolling_epoch_s={epoch_state['rolling_unit_seconds']} "
            f"elapsed={format_duration(epoch_state['elapsed_seconds'])} "
            f"ETA={format_duration(epoch_state['eta_seconds'])} "
            f"finish={epoch_state['estimated_finish_at']} "
            f"quality={epoch_state['eta_quality']}"
        )
        logger.event("epoch_end", stage=1, epoch=epoch + 1, L=tot,
                     L_region=ep_region, L_topo=ep_topo, L_simple=ep_simple,
                     grad_norm_avg=s_gn / max(n, 1), dt=ep_dt, **epoch_state)

        if tot < best:
            best = tot
            _torch_save_atomic({
                "config": asdict(cfg),
                "parent_config": parent_cfg,
                "encoder": encoder.state_dict(),
                "unet": unet.state_dict(),
                "topo_head": topo_head.state_dict(),
                "ema_encoder": ema.state_dict(),
                "best_loss": best,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "rng_state": capture_rng_state(),
                "rng_strategy": rng_strategy,
                "epoch": epoch + 1,
            }, best_ckpt_path)
            logger.info(f"  -> saved best ckpt @ ep {epoch+1} (L={best:.4f})")

        _torch_save_atomic({
            "config": asdict(cfg),
            "parent_config": parent_cfg,
            "encoder": encoder.state_dict(),
            "unet": unet.state_dict(),
            "topo_head": topo_head.state_dict(),
            "ema_encoder": ema.state_dict(),
            "ema": ema.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch + 1,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "best_loss": best,
            "rng_state": capture_rng_state(),
            "rng_strategy": rng_strategy,
        }, last_ckpt_path)

    logger.banner("Stage 1 done", {
        "best_loss": f"{best:.4f}",
        "ckpt": best_ckpt_path,
        "last_ckpt": last_ckpt_path,
        "total_time_s": f"{time.time()-t_start:.1f}",
    })
    logger.close()
    return {
        "best_loss": best,
        "ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "start_epoch": start_epoch,
        "end_epoch": cfg.epochs,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-ckpt", default=None)
    p.add_argument("--pd-json", default=None)
    p.add_argument("--base-path", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Stage1Config()
    if args.smoke:
        cfg.epochs = 1
        cfg.batch_size = 2
        cfg.num_workers = 0
        cfg.run_name = "smoke_stage1"
    for k in ("parent_ckpt", "pd_json", "base_path", "epochs", "batch_size",
              "run_name", "output_dir", "device"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    out = run_single_experiment(cfg)
    print(json.dumps(out, indent=2))
