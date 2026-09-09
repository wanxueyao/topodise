"""Logging helpers shared by all TopoDiSE stage runners."""
from __future__ import annotations
import os
import json
import time
import socket
import platform
from collections import deque
from typing import Optional, Dict, Any, Iterable

import torch


def count_params(modules) -> Dict[str, int]:
    """modules: iterable of (name, nn.Module). Returns dict with total/trainable per name + overall."""
    out: Dict[str, int] = {}
    g_total = 0
    g_train = 0
    for name, m in modules:
        if m is None:
            continue
        tot = sum(p.numel() for p in m.parameters())
        tra = sum(p.numel() for p in m.parameters() if p.requires_grad)
        out[f"{name}.total"] = tot
        out[f"{name}.trainable"] = tra
        g_total += tot
        g_train += tra
    out["overall.total"] = g_total
    out["overall.trainable"] = g_train
    return out


def grad_global_norm(params: Iterable[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return sq ** 0.5


def env_banner() -> Dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["cudnn"] = torch.backends.cudnn.version()
    return info


class TrainLogger:
    """Tee-style logger that writes to stdout and a file, with JSON event support."""

    def __init__(self, run_dir: str, log_name: str = "train.log",
                 events_name: str = "events.jsonl"):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, log_name)
        self.events_path = os.path.join(run_dir, events_name)
        self._log_f = open(self.log_path, "a", buffering=1)
        self._events_f = open(self.events_path, "a", buffering=1)
        self._t_start = time.time()

    def close(self):
        try:
            self._log_f.close()
            self._events_f.close()
        except Exception:
            pass

    def elapsed(self) -> float:
        return time.time() - self._t_start

    def info(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_f.write(line + "\n")

    def event(self, kind: str, **kwargs):
        rec = {"t": round(self.elapsed(), 3), "kind": kind, **kwargs}
        self._events_f.write(json.dumps(rec, default=str) + "\n")

    def banner(self, title: str, kv: Dict[str, Any]):
        self.info("=" * 80)
        self.info(title)
        self.info("-" * 80)
        for k, v in kv.items():
            self.info(f"  {k}: {v}")
        self.info("=" * 80)


class EMAMeter:
    """Exponential moving average for scalar streams."""

    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value


class RateMeter:
    """Rolling samples-per-second meter."""

    def __init__(self, window: int = 20):
        self.times = deque(maxlen=window)
        self.sizes = deque(maxlen=window)

    def update(self, dt: float, batch_size: int):
        self.times.append(dt)
        self.sizes.append(batch_size)

    def samples_per_sec(self) -> float:
        if not self.times:
            return 0.0
        return sum(self.sizes) / max(sum(self.times), 1e-9)
