"""Pipeline performance metrics — per-stage latency tracking."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default rolling window size for FPS calculation.
_DEFAULT_WINDOW = 120


@dataclass
class _FrameTiming:
    """Latency measurements for a single pipeline frame."""

    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    detect_ms: float = 0.0
    postprocess_ms: float = 0.0
    decide_ms: float = 0.0
    act_ms: float = 0.0
    total_ms: float = 0.0


class PipelineMetrics:
    """Tracks per-stage latency and rolling FPS.

    Usage::

        metrics = PipelineMetrics()
        metrics.begin_frame()
        # ... capture ...
        metrics.record("capture")
        # ... preprocess ...
        metrics.record("preprocess")
        # ... detect ...
        metrics.record("detect")
        # ... postprocess ...
        metrics.record("postprocess")
        # ... decide ...
        metrics.record("decide")
        # ... act ...
        metrics.record("act")
        metrics.end_frame()
    """

    def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
        self._window = window
        self._history: deque[_FrameTiming] = deque(
            maxlen=window,
        )
        self._frame_start: float = 0.0
        self._stage_start: float = 0.0
        self._current: _FrameTiming = _FrameTiming()
        self._frame_timestamps: deque[float] = deque(
            maxlen=window,
        )

    # -- recording API ------------------------------------------------

    def begin_frame(self) -> None:
        """Mark the start of a new pipeline frame."""
        now = time.perf_counter()
        self._frame_start = now
        self._stage_start = now
        self._current = _FrameTiming()

    def record(self, stage: str) -> None:
        """Record elapsed time since last checkpoint for *stage*.

        Valid stage names: ``capture``, ``preprocess``, ``detect``,
        ``postprocess``, ``decide``, ``act``.
        """
        now = time.perf_counter()
        elapsed_ms = (now - self._stage_start) * 1000.0
        attr = f"{stage}_ms"
        if hasattr(self._current, attr):
            setattr(self._current, attr, elapsed_ms)
        self._stage_start = now

    def end_frame(self) -> None:
        """Finalise the current frame timing and store it."""
        now = time.perf_counter()
        self._current.total_ms = (now - self._frame_start) * 1000.0
        self._history.append(self._current)
        self._frame_timestamps.append(now)

    # -- query API ----------------------------------------------------

    def rolling_fps(self) -> float:
        """Return FPS over the last *window* frames.

        Returns ``0.0`` when fewer than two frames have been
        recorded.
        """
        stamps = self._frame_timestamps
        if len(stamps) < 2:
            return 0.0
        elapsed = stamps[-1] - stamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(stamps) - 1) / elapsed

    def summary(self) -> dict[str, float]:
        """Return a dict of average latencies across stored frames.

        Keys: ``capture_ms``, ``preprocess_ms``, ``detect_ms``,
        ``postprocess_ms``, ``decide_ms``, ``act_ms``,
        ``total_ms``, ``fps``.
        """
        if not self._history:
            return {
                "capture_ms": 0.0,
                "preprocess_ms": 0.0,
                "detect_ms": 0.0,
                "postprocess_ms": 0.0,
                "decide_ms": 0.0,
                "act_ms": 0.0,
                "total_ms": 0.0,
                "fps": 0.0,
            }
        n = len(self._history)
        totals = _FrameTiming()
        for ft in self._history:
            totals.capture_ms += ft.capture_ms
            totals.preprocess_ms += ft.preprocess_ms
            totals.detect_ms += ft.detect_ms
            totals.postprocess_ms += ft.postprocess_ms
            totals.decide_ms += ft.decide_ms
            totals.act_ms += ft.act_ms
            totals.total_ms += ft.total_ms

        return {
            "capture_ms": totals.capture_ms / n,
            "preprocess_ms": totals.preprocess_ms / n,
            "detect_ms": totals.detect_ms / n,
            "postprocess_ms": totals.postprocess_ms / n,
            "decide_ms": totals.decide_ms / n,
            "act_ms": totals.act_ms / n,
            "total_ms": totals.total_ms / n,
            "fps": self.rolling_fps(),
        }

    def log_summary(self) -> None:
        """Log the current summary at INFO level."""
        s = self.summary()
        logger.info(
            "cap=%.0f pre=%.0f det=%.0f post=%.0f " "dec=%.0f act=%.0f tot=%.0fms fps=%.1f",
            s["capture_ms"],
            s["preprocess_ms"],
            s["detect_ms"],
            s["postprocess_ms"],
            s["decide_ms"],
            s["act_ms"],
            s["total_ms"],
            s["fps"],
        )
