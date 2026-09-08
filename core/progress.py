"""Small dependency-free progress and ETA helpers for experiment runners."""
from __future__ import annotations

import statistics
import time
from collections import deque
from datetime import datetime, timedelta


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class ETAEstimator:
    """Rolling-median ETA that becomes useful after only a few completed units."""

    def __init__(
        self,
        total_units: int,
        *,
        initial_completed: int = 0,
        window: int = 5,
    ):
        if total_units <= 0:
            raise ValueError("total_units must be positive")
        if not 0 <= initial_completed <= total_units:
            raise ValueError("initial_completed must be within [0, total_units]")
        self.total_units = int(total_units)
        self.last_completed = int(initial_completed)
        self.samples: deque[float] = deque(maxlen=max(1, int(window)))
        self.started = time.perf_counter()
        self.last_update = self.started

    def update(
        self,
        completed_units: int,
        *,
        interval_seconds: float | None = None,
    ) -> dict:
        completed_units = int(completed_units)
        if not self.last_completed <= completed_units <= self.total_units:
            raise ValueError(
                f"completed_units must be monotonic and <= total: "
                f"{completed_units}, last={self.last_completed}, total={self.total_units}"
            )
        now = time.perf_counter()
        delta_units = completed_units - self.last_completed
        measured = now - self.last_update if interval_seconds is None else float(interval_seconds)
        if delta_units > 0 and measured >= 0:
            self.samples.append(measured / delta_units)
        self.last_completed = completed_units
        self.last_update = now

        median_unit_seconds = statistics.median(self.samples) if self.samples else None
        remaining_units = self.total_units - completed_units
        eta_seconds = (
            median_unit_seconds * remaining_units
            if median_unit_seconds is not None else None
        )
        elapsed_seconds = now - self.started
        finish_at = (
            (datetime.now().astimezone() + timedelta(seconds=eta_seconds)).isoformat(timespec="seconds")
            if eta_seconds is not None else None
        )
        sample_count = len(self.samples)
        if sample_count < 3:
            quality = "warming_up"
        elif sample_count < self.samples.maxlen:
            quality = "usable"
        else:
            quality = "rolling_stable"
        return {
            "completed_units": completed_units,
            "total_units": self.total_units,
            "remaining_units": remaining_units,
            "progress_pct": round(100.0 * completed_units / self.total_units, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "rolling_unit_seconds": (
                round(median_unit_seconds, 3) if median_unit_seconds is not None else None
            ),
            "units_per_second": (
                round(1.0 / median_unit_seconds, 6)
                if median_unit_seconds is not None and median_unit_seconds > 0 else None
            ),
            "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
            "estimated_total_session_seconds": (
                round(elapsed_seconds + eta_seconds, 3) if eta_seconds is not None else None
            ),
            "estimated_finish_at": finish_at,
            "eta_quality": quality,
            "eta_sample_count": sample_count,
        }
