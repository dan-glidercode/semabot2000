"""Thorough tests for core domain models."""

from __future__ import annotations

import pytest

from semabot.core.models import Action, BoundingBox, Detection, GameState

# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_center_calculation(self) -> None:
        box = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        assert box.center == (20.0, 30.0)

    def test_center_with_non_integer_coords(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=5.0, y2=7.0)
        assert box.center == (2.5, 3.5)

    def test_width(self) -> None:
        box = BoundingBox(x1=10.0, y1=0.0, x2=50.0, y2=30.0)
        assert box.width == 40.0

    def test_height(self) -> None:
        box = BoundingBox(x1=0.0, y1=10.0, x2=30.0, y2=60.0)
        assert box.height == 50.0

    def test_area(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)
        assert box.area == 200.0

    def test_zero_size_box(self) -> None:
        box = BoundingBox(x1=5.0, y1=5.0, x2=5.0, y2=5.0)
        assert box.center == (5.0, 5.0)
        assert box.width == 0.0
        assert box.height == 0.0
        assert box.area == 0.0

    def test_frozen_immutability(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        with pytest.raises(AttributeError):
            box.x1 = 99.0  # type: ignore[misc]

    def test_frozen_immutability_all_fields(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        with pytest.raises(AttributeError):
            box.y1 = 99.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            box.x2 = 99.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            box.y2 = 99.0  # type: ignore[misc]

    def test_equality(self) -> None:
        a = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        b = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert a == b

    def test_inequality(self) -> None:
        a = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        b = BoundingBox(x1=0.0, y1=2.0, x2=3.0, y2=4.0)
        assert a != b


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetection:
    """Tests for Detection dataclass."""

    def test_construction(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        det = Detection(class_name="person", confidence=0.95, bbox=bbox)
        assert det.class_name == "person"
        assert det.confidence == 0.95
        assert det.bbox is bbox

    def test_immutability(self) -> None:
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        det = Detection(class_name="cat", confidence=0.8, bbox=bbox)
        with pytest.raises(AttributeError):
            det.class_name = "dog"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            det.confidence = 0.1  # type: ignore[misc]
        with pytest.raises(AttributeError):
            det.bbox = bbox  # type: ignore[misc]

    def test_equality(self) -> None:
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        a = Detection(class_name="car", confidence=0.7, bbox=bbox)
        b = Detection(class_name="car", confidence=0.7, bbox=bbox)
        assert a == b


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------


class TestGameState:
    """Tests for GameState dataclass."""

    def test_empty_detections(self) -> None:
        state = GameState(
            detections=(),
            frame_width=1920,
            frame_height=1080,
            timestamp=1.0,
        )
        assert state.detections == ()
        assert state.frame_width == 1920
        assert state.frame_height == 1080
        assert state.timestamp == 1.0

    def test_populated_detections(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        det = Detection(class_name="person", confidence=0.9, bbox=bbox)
        state = GameState(
            detections=(det,),
            frame_width=1280,
            frame_height=720,
            timestamp=2.5,
        )
        assert len(state.detections) == 1
        assert state.detections[0].class_name == "person"

    def test_multiple_detections(self) -> None:
        d1 = Detection(
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
        )
        d2 = Detection(
            class_name="car",
            confidence=0.7,
            bbox=BoundingBox(50.0, 50.0, 100.0, 100.0),
        )
        state = GameState(
            detections=(d1, d2),
            frame_width=640,
            frame_height=480,
            timestamp=3.0,
        )
        assert len(state.detections) == 2

    def test_immutability(self) -> None:
        state = GameState(
            detections=(),
            frame_width=1920,
            frame_height=1080,
            timestamp=0.0,
        )
        with pytest.raises(AttributeError):
            state.detections = ()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            state.frame_width = 640  # type: ignore[misc]
        with pytest.raises(AttributeError):
            state.timestamp = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class TestAction:
    """Tests for Action dataclass."""

    def test_default_empty(self) -> None:
        action = Action()
        assert action.keys_press == ()
        assert action.keys_release == ()
        assert action.description == ""

    def test_with_keys(self) -> None:
        action = Action(
            keys_press=("w", "a"),
            keys_release=("d",),
            description="move forward-left",
        )
        assert action.keys_press == ("w", "a")
        assert action.keys_release == ("d",)
        assert action.description == "move forward-left"

    def test_immutability(self) -> None:
        action = Action(keys_press=("w",))
        with pytest.raises(AttributeError):
            action.keys_press = ("s",)  # type: ignore[misc]
        with pytest.raises(AttributeError):
            action.keys_release = ("d",)  # type: ignore[misc]
        with pytest.raises(AttributeError):
            action.description = "nope"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = Action(keys_press=("w",), description="go")
        b = Action(keys_press=("w",), description="go")
        assert a == b
