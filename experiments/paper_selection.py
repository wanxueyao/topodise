"""读取论文最终采用的实验配置。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_paper_experiments(method: str) -> list[dict[str, Any]]:
    """读取某个基线方法按任务选定的最佳配置。"""
    manifest_path = Path(__file__).with_name("paper_baseline_experiments.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if method not in manifest:
        raise KeyError(f"Unknown paper baseline method: {method}")
    return list(manifest[method])
