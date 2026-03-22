"""Integration test: all pipeline components with synthetic data."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np

from semabot.action.null_controller import NullInputController
from semabot.app.orchestrator import BotOrchestrator
from semabot.core.models import BoundingBox, Detection
from semabot.intelligence.preprocessor import YoloPreprocessor
from semabot.intelligence.state_builder import GameStateBuilder

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _make_config() -> MagicMock:
    """Return a minimal mock BotConfig."""
    config = MagicMock()
    config.target_fps = 25
    config.detection.model_path = "fake.onnx"
    config.detection.provider = "CPUExecutionProvider"
    config.detection.confidence_threshold = 0.35
    config.detection.input_size = 640
    config.capture.method = "mock"
    config.capture.window_title = "Test"
    config.log_level = "DEBUG"
    return config


def _make_game_profile() -> MagicMock:
    """Return a minimal mock GameProfile."""
    profile = MagicMock()
    profile.name = "test_game"
    profile.detection.target_classes = ("person",)
    profile.detection.target_min_confidence = 0.4
    profile.controls.move_forward = "w"
    profile.controls.move_backward = "s"
    profile.controls.move_left = "a"
    profile.controls.move_right = "d"
    profile.controls.camera_left = "left"
    profile.controls.camera_right = "right"
    profile.controls.jump = "space"
    profile.controls.interact = "e"
    profile.behavior.approach_tolerance_px = 60
    profile.behavior.close_enough_bbox_height = 200
    profile.behavior.wander_cycle_ticks = 30
    return profile


class _MockFrameSource:
    """Frame source that returns a pre-built numpy frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def start(self) -> None:
        pass

    def get_latest_frame(self) -> np.ndarray:
        return self._frame

    def stop(self) -> None:
        pass


class _MockDetector:
    """Detector that returns pre-configured detections."""

    def __init__(
        self,
        detections: list[Detection],
    ) -> None:
        self._detections = detections

    def detect(self, blob: np.ndarray) -> list[Detection]:
        return self._detections


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------


class TestFullPipeline:
    """All components wired together with synthetic data."""

    def test_orchestrator_runs_without_errors(self) -> None:
        """Bot orchestrator runs the full pipeline for 0.1s."""
        frame = np.random.randint(
            0,
            256,
            (600, 800, 3),
            dtype=np.uint8,
        )
        known_detections = [
            Detection(
                class_name="person",
                confidence=0.85,
                bbox=BoundingBox(
                    x1=100.0,
                    y1=100.0,
                    x2=200.0,
                    y2=300.0,
                ),
            ),
        ]

        frame_source = _MockFrameSource(frame)
        preprocessor = YoloPreprocessor(input_size=640)
        detector = _MockDetector(known_detections)
        state_builder = GameStateBuilder(
            confidence_threshold=0.35,
        )

        # Use real behavior tree engine
        from semabot.intelligence.behavior.engine import (
            BehaviorTreeEngine,
        )
        from semabot.intelligence.behavior.trees import (
            build_steal_a_brainrot_tree,
        )

        profile = _make_game_profile()
        tree = build_steal_a_brainrot_tree(profile)
        decision_engine = BehaviorTreeEngine(tree)

        input_controller = NullInputController()
        config = _make_config()

        orchestrator = BotOrchestrator(
            frame_source=frame_source,
            preprocessor=preprocessor,
            detector=detector,
            state_builder=state_builder,
            decision_engine=decision_engine,
            input_controller=input_controller,
            config=config,
        )

        orchestrator.run(duration=0.1)

        # If we got here without exception, the pipeline works.

    def test_null_controller_logs_actions(self) -> None:
        """NullInputController logs actions during the run."""
        frame = np.random.randint(
            0,
            256,
            (600, 800, 3),
            dtype=np.uint8,
        )
        known_detections = [
            Detection(
                class_name="person",
                confidence=0.85,
                bbox=BoundingBox(
                    x1=350.0,
                    y1=100.0,
                    x2=450.0,
                    y2=300.0,
                ),
            ),
        ]

        frame_source = _MockFrameSource(frame)
        preprocessor = YoloPreprocessor(input_size=640)
        detector = _MockDetector(known_detections)
        state_builder = GameStateBuilder(
            confidence_threshold=0.35,
        )

        from semabot.intelligence.behavior.engine import (
            BehaviorTreeEngine,
        )
        from semabot.intelligence.behavior.trees import (
            build_steal_a_brainrot_tree,
        )

        profile = _make_game_profile()
        tree = build_steal_a_brainrot_tree(profile)
        decision_engine = BehaviorTreeEngine(tree)

        input_controller = NullInputController()
        config = _make_config()

        orchestrator = BotOrchestrator(
            frame_source=frame_source,
            preprocessor=preprocessor,
            detector=detector,
            state_builder=state_builder,
            decision_engine=decision_engine,
            input_controller=input_controller,
            config=config,
        )

        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        nc_logger = logging.getLogger(
            "semabot.action.null_controller",
        )
        nc_logger.setLevel(logging.DEBUG)
        collector = _Collector()
        nc_logger.addHandler(collector)
        try:
            orchestrator.run(duration=0.1)
        finally:
            nc_logger.removeHandler(collector)

        execute_records = [r for r in records if "NullInputController.execute" in r.getMessage()]
        assert len(execute_records) > 0

    def test_no_detections_wanders(self) -> None:
        """With no detections the behavior tree wanders."""
        frame = np.random.randint(
            0,
            256,
            (600, 800, 3),
            dtype=np.uint8,
        )

        frame_source = _MockFrameSource(frame)
        preprocessor = YoloPreprocessor(input_size=640)
        detector = _MockDetector([])
        state_builder = GameStateBuilder(
            confidence_threshold=0.35,
        )

        from semabot.intelligence.behavior.engine import (
            BehaviorTreeEngine,
        )
        from semabot.intelligence.behavior.trees import (
            build_steal_a_brainrot_tree,
        )

        profile = _make_game_profile()
        tree = build_steal_a_brainrot_tree(profile)
        decision_engine = BehaviorTreeEngine(tree)

        input_controller = NullInputController()
        config = _make_config()

        orchestrator = BotOrchestrator(
            frame_source=frame_source,
            preprocessor=preprocessor,
            detector=detector,
            state_builder=state_builder,
            decision_engine=decision_engine,
            input_controller=input_controller,
            config=config,
        )

        # Should complete without errors even with no detections
        orchestrator.run(duration=0.1)
