"""Tests for PipelineMetrics."""

from __future__ import annotations

import logging
import time

from semabot.app.metrics import PipelineMetrics

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

_STAGES = (
    "capture",
    "preprocess",
    "detect",
    "postprocess",
    "decide",
    "act",
)


def _record_fake_frame(
    metrics: PipelineMetrics,
) -> None:
    """Simulate one pipeline frame with tiny sleeps."""
    metrics.begin_frame()
    for stage in _STAGES:
        time.sleep(0.001)
        metrics.record(stage)
    metrics.end_frame()


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------


class TestPipelineMetrics:
    """Core metrics tracking and summarisation."""

    def test_summary_empty(self) -> None:
        """Summary with no frames returns all zeros."""
        m = PipelineMetrics()
        s = m.summary()
        assert s["capture_ms"] == 0.0
        assert s["total_ms"] == 0.0
        assert s["fps"] == 0.0

    def test_summary_keys(self) -> None:
        """Summary dict contains all expected keys."""
        m = PipelineMetrics()
        _record_fake_frame(m)
        s = m.summary()
        expected = {
            "capture_ms",
            "preprocess_ms",
            "detect_ms",
            "postprocess_ms",
            "decide_ms",
            "act_ms",
            "total_ms",
            "fps",
        }
        assert set(s.keys()) == expected

    def test_total_ms_positive_after_frame(self) -> None:
        """After recording a frame, total_ms is positive."""
        m = PipelineMetrics()
        _record_fake_frame(m)
        s = m.summary()
        assert s["total_ms"] > 0

    def test_stage_times_positive(self) -> None:
        """Each recorded stage has positive latency."""
        m = PipelineMetrics()
        _record_fake_frame(m)
        s = m.summary()
        for stage in _STAGES:
            assert s[f"{stage}_ms"] > 0

    def test_rolling_fps_zero_with_few_frames(self) -> None:
        """rolling_fps returns 0 with fewer than 2 frames."""
        m = PipelineMetrics()
        assert m.rolling_fps() == 0.0

        m.begin_frame()
        m.end_frame()
        assert m.rolling_fps() == 0.0

    def test_rolling_fps_positive_after_frames(self) -> None:
        """rolling_fps is positive after multiple frames."""
        m = PipelineMetrics()
        for _ in range(5):
            _record_fake_frame(m)
        assert m.rolling_fps() > 0

    def test_summary_averages_multiple_frames(self) -> None:
        """Summary averages across multiple recorded frames."""
        m = PipelineMetrics()
        for _ in range(10):
            _record_fake_frame(m)
        s = m.summary()
        # Average total should be less than total of all frames
        assert s["total_ms"] > 0
        assert s["total_ms"] < 1000.0

    def test_window_limits_history(self) -> None:
        """History is bounded by the window parameter."""
        m = PipelineMetrics(window=5)
        for _ in range(20):
            _record_fake_frame(m)
        # Internal history should be capped at 5
        assert len(m._history) == 5

    def test_log_summary(self) -> None:
        """log_summary emits an INFO-level log message."""
        m = PipelineMetrics()
        _record_fake_frame(m)

        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        metrics_logger = logging.getLogger(
            "semabot.app.metrics",
        )
        collector = _Collector()
        metrics_logger.addHandler(collector)
        metrics_logger.setLevel(logging.DEBUG)
        try:
            m.log_summary()
        finally:
            metrics_logger.removeHandler(collector)

        info_records = [r for r in records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        msg = info_records[0].getMessage()
        assert "det=" in msg or "Pipeline metrics" in msg

    def test_begin_frame_resets_current(self) -> None:
        """begin_frame starts a fresh timing record."""
        m = PipelineMetrics()
        _record_fake_frame(m)
        m.begin_frame()
        # After begin but before end, current is fresh
        m.record("capture")
        m.end_frame()
        assert len(m._history) == 2

    def test_record_unknown_stage_ignored(self) -> None:
        """Recording an unknown stage name does not crash."""
        m = PipelineMetrics()
        m.begin_frame()
        m.record("nonexistent_stage")
        m.end_frame()
        s = m.summary()
        assert s["total_ms"] >= 0
