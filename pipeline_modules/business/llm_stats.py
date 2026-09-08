"""Per-branch accounting of LLM calls made during one pipeline run."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

ZONE_CHECK = "zone_check"
ZONE_CHECK_DEEP = "zone_check_deep"
NOT_ALLOWED_RERANK = "not_allowed_rerank"


@dataclass
class BranchStats:
    calls: int = 0
    failures: int = 0
    seconds: float = 0.0


_lock = threading.Lock()
_branches: dict[str, BranchStats] = {}


def reset() -> None:
    with _lock:
        _branches.clear()


def snapshot() -> dict[str, BranchStats]:
    with _lock:
        return {name: BranchStats(**vars(stats)) for name, stats in _branches.items()}


@contextmanager
def record(branch: str) -> Iterator[None]:
    started = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.perf_counter() - started
        with _lock:
            stats = _branches.setdefault(branch, BranchStats())
            stats.calls += 1
            stats.seconds += elapsed
            if failed:
                stats.failures += 1


def format_summary() -> str:
    """Render the run totals. Seconds are summed across worker threads, so they
    exceed wall time whenever calls run in parallel."""
    taken = snapshot()
    if not taken:
        return "calls=0"
    total_seconds = sum(stats.seconds for stats in taken.values()) or 1.0
    parts = [
        f"{name}: calls={stats.calls} failures={stats.failures} "
        f"busy_s={stats.seconds:.1f} share={100 * stats.seconds / total_seconds:.0f}% "
        f"mean_s={stats.seconds / stats.calls:.2f}"
        for name, stats in sorted(taken.items(), key=lambda item: -item[1].seconds)
        if stats.calls
    ]
    totals = (
        f"total: calls={sum(s.calls for s in taken.values())} "
        f"failures={sum(s.failures for s in taken.values())} "
        f"busy_s={sum(s.seconds for s in taken.values()):.1f}"
    )
    return " | ".join([*parts, totals])
