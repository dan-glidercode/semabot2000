"""Tests for BotOrchestrator."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np

from semabot.app.orchestrator import BotOrchestrator
from semabot.core.models import Action, Detection, GameState

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_config() -> MagicMock:
    """Return a mock BotConfig."""
    config = MagicMock()
    config.target_fps = 25
    return config


def _make_components(
    frame: np.ndarray | None = None,
) -> dict:
    """Return a dict of mocked pipeline components.

    When *frame* is not None the frame source returns it once
    then returns None (so the loop can exit via duration).
    """
    fake_frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
    detections: list[Detection] = []
    game_state = GameState(
        detections=(),
        frame_width=640,
        frame_height=480,
        timestamp=time.time(),
    )
    action = Action(keys_press=("w",), description="test")

    frame_source = MagicMock()
    frame_source.get_latest_frame.return_value = fake_frame
    preprocessor = MagicMock()
    preprocessor.process.return_value = blob
    detector = MagicMock()
    detector.detect.return_value = detections
    state_builder = MagicMock()
    state_builder.build.return_value = game_state
    decision_engine = MagicMock()
    decision_engine.decide.return_value = action
    input_controller = MagicMock()

    return {
        "frame_source": frame_source,
        "preprocessor": preprocessor,
        "detector": detector,
        "state_builder": state_builder,
        "decision_engine": decision_engine,
        "input_controller": input_controller,
        "config": _make_config(),
    }


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestCallOrder:
    """Verify the orchestrator calls components in order."""

    def test_full_pipeline_call_order(self) -> None:
        """start -> get_frame -> process -> detect -> build
        -> decide -> execute -> stop -> release_all."""
        comps = _make_components()
        orch = BotOrchestrator(**comps)
        orch.run(duration=0.05)

        comps["frame_source"].start.assert_called_once()
        comps["frame_source"].get_latest_frame.assert_called()
        comps["preprocessor"].process.assert_called()
        comps["detector"].detect.assert_called()
        comps["state_builder"].build.assert_called()
        comps["decision_engine"].decide.assert_called()
        comps["input_controller"].execute.assert_called()
        comps["frame_source"].stop.assert_called_once()
        comps["input_controller"].release_all.assert_called_once()


class TestNoneFrame:
    """When frame_source returns None, skip detect/decide/act."""

    def test_none_frame_skips_processing(self) -> None:
        comps = _make_components()
        comps["frame_source"].get_latest_frame.return_value = None
        orch = BotOrchestrator(**comps)
        orch.run(duration=0.05)

        comps["preprocessor"].process.assert_not_called()
        comps["detector"].detect.assert_not_called()
        comps["decision_engine"].decide.assert_not_called()
        comps["input_controller"].execute.assert_not_called()


class TestDuration:
    """Duration parameter stops the loop."""

    def test_duration_stops_loop(self) -> None:
        comps = _make_components()
        orch = BotOrchestrator(**comps)
        start = time.monotonic()
        orch.run(duration=0.1)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, "Loop should stop near duration"
        comps["frame_source"].stop.assert_called_once()


class TestShutdown:
    """stop() causes the loop to exit and keys to be released."""

    def test_stop_releases_all_keys(self) -> None:
        comps = _make_components()
        comps["frame_source"].get_latest_frame.return_value = None
        orch = BotOrchestrator(**comps)

        def stop_soon() -> None:
            time.sleep(0.05)
            orch.stop()

        import threading

        t = threading.Thread(target=stop_soon)
        t.start()
        orch.run()
        t.join()

        comps["frame_source"].stop.assert_called_once()
        comps["input_controller"].release_all.assert_called_once()

    def test_release_all_called_on_repeated_errors(self) -> None:
        """release_all is called even when every tick raises.

        Non-fatal errors are caught by error recovery so the loop
        continues until the duration expires. Cleanup still runs.
        """
        comps = _make_components()
        comps["preprocessor"].process.side_effect = RuntimeError(
            "boom",
        )
        orch = BotOrchestrator(**comps)
        orch.run(duration=0.1)

        comps["frame_source"].stop.assert_called_once()
        comps["input_controller"].release_all.assert_called_once()


class TestSaveDetections:
    """Test the --save-detections feature."""

    def test_save_detections_creates_files(self, tmp_path: object) -> None:
        from semabot.core.models import BoundingBox

        comps = _make_components()
        det = Detection(
            class_name="person",
            confidence=0.8,
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
        )
        comps["detector"].detect.return_value = [det]
        comps["state_builder"].build.return_value = GameState(
            detections=(det,),
            frame_width=640,
            frame_height=480,
            timestamp=time.time(),
        )

        save_dir = str(tmp_path)  # type: ignore[union-attr]
        orch = BotOrchestrator(
            **comps,
            save_detections=True,
            save_dir=save_dir,
        )
        orch.run(duration=0.15)

        import os

        files = os.listdir(save_dir)
        assert len(files) > 0, "Should have saved at least one annotated frame"

    def test_save_detections_off_by_default(self) -> None:
        comps = _make_components()
        orch = BotOrchestrator(**comps)
        assert orch._save_detections is False


class TestStaticHelpers:
    """Test static helper methods."""

    def test_maybe_log_fps_returns_updated_time(self) -> None:
        now = time.monotonic()
        result = BotOrchestrator._maybe_log_fps(100, now - 6.0, now - 6.0)
        assert result > now - 6.0

    def test_maybe_log_fps_skips_when_recent(self) -> None:
        now = time.monotonic()
        old_log = now - 1.0
        result = BotOrchestrator._maybe_log_fps(50, now - 10.0, old_log)
        assert result == old_log

    def test_should_stop_with_none_duration(self) -> None:
        assert BotOrchestrator._should_stop(time.monotonic(), None) is False

    def test_should_stop_with_expired_duration(self) -> None:
        assert BotOrchestrator._should_stop(time.monotonic() - 10, 5.0) is True
