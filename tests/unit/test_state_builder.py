"""Tests for GameStateBuilder."""

from __future__ import annotations

import time

import numpy as np
import pytest

from semabot.core.models import BoundingBox, Detection, GameState
from semabot.intelligence.state_builder import GameStateBuilder


def _det(
    name: str = "person",
    conf: float = 0.9,
) -> Detection:
    """Shorthand factory for a Detection with dummy bbox."""
    return Detection(
        class_name=name,
        confidence=conf,
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
    )


class TestGameStateBuilder:
    """Verify filtering, dimensions, timestamp, and immutability."""

    def test_filters_below_threshold(self) -> None:
        builder = GameStateBuilder(confidence_threshold=0.5)
        detections = [
            _det("person", 0.9),
            _det("car", 0.3),
            _det("dog", 0.6),
        ]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = builder.build(detections, frame)

        names = [d.class_name for d in state.detections]
        assert "person" in names
        assert "dog" in names
        assert "car" not in names
        assert len(state.detections) == 2

    def test_keeps_at_threshold(self) -> None:
        """Detection exactly at the threshold is kept."""
        builder = GameStateBuilder(confidence_threshold=0.5)
        detections = [_det("cat", 0.5)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = builder.build(detections, frame)

        assert len(state.detections) == 1
        assert state.detections[0].class_name == "cat"

    def test_empty_detections(self) -> None:
        builder = GameStateBuilder()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = builder.build([], frame)

        assert state.detections == ()
        assert state.frame_width == 1280
        assert state.frame_height == 720

    def test_all_filtered(self) -> None:
        """All detections below threshold results in empty tuple."""
        builder = GameStateBuilder(confidence_threshold=0.8)
        detections = [_det("a", 0.1), _det("b", 0.2)]
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        state = builder.build(detections, frame)

        assert state.detections == ()

    def test_frame_dimensions_extracted(self) -> None:
        builder = GameStateBuilder()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        state = builder.build([], frame)

        assert state.frame_width == 1920
        assert state.frame_height == 1080

    def test_frame_dimensions_non_standard(self) -> None:
        """Non-standard frame shape is handled correctly."""
        builder = GameStateBuilder()
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        state = builder.build([], frame)

        assert state.frame_width == 500
        assert state.frame_height == 300

    def test_timestamp_is_set(self) -> None:
        builder = GameStateBuilder()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        before = time.time()
        state = builder.build([], frame)
        after = time.time()

        assert state.timestamp > 0
        assert before <= state.timestamp <= after

    def test_result_is_immutable(self) -> None:
        builder = GameStateBuilder()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = builder.build([], frame)

        assert isinstance(state, GameState)
        with pytest.raises(AttributeError):
            state.frame_width = 999  # type: ignore[misc]
        with pytest.raises(AttributeError):
            state.detections = ()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            state.timestamp = 0.0  # type: ignore[misc]

    def test_detections_tuple_type(self) -> None:
        """Returned detections field is a tuple, not a list."""
        builder = GameStateBuilder()
        detections = [_det("person", 0.9)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = builder.build(detections, frame)

        assert isinstance(state.detections, tuple)

    def test_default_threshold(self) -> None:
        """Default confidence_threshold is 0.3."""
        builder = GameStateBuilder()
        detections = [
            _det("person", 0.3),
            _det("car", 0.29),
        ]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = builder.build(detections, frame)

        assert len(state.detections) == 1
        assert state.detections[0].class_name == "person"
