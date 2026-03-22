"""Tests for behavior-tree condition nodes."""

from __future__ import annotations

import py_trees
import pytest

from semabot.core.models import (
    Action,
    BoundingBox,
    Detection,
    GameState,
)
from semabot.intelligence.behavior.conditions import (
    HasDetection,
    TargetClose,
    TargetInCenter,
)

SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE


# ------------------------------------------------------------------ helpers
def _make_state(
    detections: tuple[Detection, ...] = (),
    width: int = 640,
    height: int = 640,
) -> GameState:
    return GameState(
        detections=detections,
        frame_width=width,
        frame_height=height,
        timestamp=0.0,
    )


def _det(
    cls: str = "person",
    conf: float = 0.7,
    x1: float = 0,
    y1: float = 0,
    x2: float = 100,
    y2: float = 100,
) -> Detection:
    return Detection(
        class_name=cls,
        confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _setup_blackboard(state: GameState) -> None:
    """Write *state* to the global blackboard."""
    py_trees.blackboard.Blackboard.enable_activity_stream()
    bb = py_trees.blackboard.Client(name="test_setup")
    bb.register_key(
        key="game_state",
        access=py_trees.common.Access.WRITE,
    )
    bb.register_key(
        key="action",
        access=py_trees.common.Access.WRITE,
    )
    bb.game_state = state
    bb.action = Action()


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    """Reset blackboard between every test."""
    py_trees.blackboard.Blackboard.enable_activity_stream()
    # Provide a clean slate.
    _setup_blackboard(_make_state())


# ---------------------------------------------------------------- HasDetection


class TestHasDetection:
    """Tests for HasDetection condition node."""

    def test_success_when_matching_detection(self) -> None:
        state = _make_state((_det(),))
        _setup_blackboard(state)
        node = HasDetection("check", class_name="person")
        node.setup()
        assert node.update() == SUCCESS

    def test_failure_when_no_detections(self) -> None:
        _setup_blackboard(_make_state())
        node = HasDetection("check", class_name="person")
        node.setup()
        assert node.update() == FAILURE

    def test_failure_when_wrong_class(self) -> None:
        state = _make_state((_det(cls="car"),))
        _setup_blackboard(state)
        node = HasDetection("check", class_name="person")
        node.setup()
        assert node.update() == FAILURE

    def test_failure_when_confidence_too_low(self) -> None:
        state = _make_state((_det(conf=0.1),))
        _setup_blackboard(state)
        node = HasDetection(
            "check",
            class_name="person",
            min_confidence=0.3,
        )
        node.setup()
        assert node.update() == FAILURE

    def test_success_at_exact_confidence_threshold(self) -> None:
        state = _make_state((_det(conf=0.3),))
        _setup_blackboard(state)
        node = HasDetection(
            "check",
            class_name="person",
            min_confidence=0.3,
        )
        node.setup()
        assert node.update() == SUCCESS

    def test_multiple_detections_one_match(self) -> None:
        state = _make_state(
            (
                _det(cls="car", conf=0.9),
                _det(cls="person", conf=0.5),
            )
        )
        _setup_blackboard(state)
        node = HasDetection("check", class_name="person")
        node.setup()
        assert node.update() == SUCCESS


# ------------------------------------------------------------- TargetInCenter


class TestTargetInCenter:
    """Tests for TargetInCenter condition node."""

    def test_success_when_centered(self) -> None:
        # center_x = 320 for a 640-wide frame; detection at 300-340
        state = _make_state((_det(x1=300, y1=0, x2=340, y2=100),))
        _setup_blackboard(state)
        node = TargetInCenter(
            "center",
            class_name="person",
            tolerance_px=60,
        )
        node.setup()
        assert node.update() == SUCCESS

    def test_failure_when_far_left(self) -> None:
        state = _make_state((_det(x1=0, y1=0, x2=100, y2=100),))
        _setup_blackboard(state)
        node = TargetInCenter(
            "center",
            class_name="person",
            tolerance_px=60,
        )
        node.setup()
        assert node.update() == FAILURE

    def test_failure_when_far_right(self) -> None:
        state = _make_state((_det(x1=540, y1=0, x2=640, y2=100),))
        _setup_blackboard(state)
        node = TargetInCenter(
            "center",
            class_name="person",
            tolerance_px=60,
        )
        node.setup()
        assert node.update() == FAILURE

    def test_failure_when_no_detections(self) -> None:
        _setup_blackboard(_make_state())
        node = TargetInCenter(
            "center",
            class_name="person",
        )
        node.setup()
        assert node.update() == FAILURE

    def test_edge_exactly_at_tolerance(self) -> None:
        # frame center = 320, tolerance = 60 -> 260..380
        # det center_x = 260 -> offset = |260-320| = 60 <= 60
        state = _make_state((_det(x1=250, y1=0, x2=270, y2=100),))
        _setup_blackboard(state)
        node = TargetInCenter(
            "center",
            class_name="person",
            tolerance_px=60,
        )
        node.setup()
        assert node.update() == SUCCESS

    def test_picks_highest_confidence(self) -> None:
        # Low-conf detection is centered, high-conf is off-center.
        far_left = _det(conf=0.9, x1=0, y1=0, x2=60, y2=100)
        centered = _det(conf=0.3, x1=300, y1=0, x2=340, y2=100)
        state = _make_state((far_left, centered))
        _setup_blackboard(state)
        node = TargetInCenter(
            "center",
            class_name="person",
            tolerance_px=60,
        )
        node.setup()
        # best match = far_left (0.9) which is NOT centered
        assert node.update() == FAILURE


# --------------------------------------------------------------- TargetClose


class TestTargetClose:
    """Tests for TargetClose condition node."""

    def test_success_when_close(self) -> None:
        # bbox height = 400
        state = _make_state((_det(x1=0, y1=100, x2=100, y2=500),))
        _setup_blackboard(state)
        node = TargetClose(
            "close",
            class_name="person",
            min_bbox_height=200,
        )
        node.setup()
        assert node.update() == SUCCESS

    def test_failure_when_far(self) -> None:
        # bbox height = 50
        state = _make_state((_det(x1=0, y1=0, x2=100, y2=50),))
        _setup_blackboard(state)
        node = TargetClose(
            "close",
            class_name="person",
            min_bbox_height=200,
        )
        node.setup()
        assert node.update() == FAILURE

    def test_failure_when_no_detections(self) -> None:
        _setup_blackboard(_make_state())
        node = TargetClose(
            "close",
            class_name="person",
        )
        node.setup()
        assert node.update() == FAILURE

    def test_edge_exactly_at_threshold(self) -> None:
        # bbox height = 200 exactly
        state = _make_state((_det(x1=0, y1=0, x2=100, y2=200),))
        _setup_blackboard(state)
        node = TargetClose(
            "close",
            class_name="person",
            min_bbox_height=200,
        )
        node.setup()
        assert node.update() == SUCCESS

    def test_failure_when_wrong_class(self) -> None:
        state = _make_state((_det(cls="car", x1=0, y1=0, x2=100, y2=500),))
        _setup_blackboard(state)
        node = TargetClose(
            "close",
            class_name="person",
            min_bbox_height=200,
        )
        node.setup()
        assert node.update() == FAILURE

    def test_low_confidence_still_matches(self) -> None:
        """TargetClose has no confidence filter of its own."""
        state = _make_state((_det(conf=0.1, x1=0, y1=0, x2=100, y2=500),))
        _setup_blackboard(state)
        node = TargetClose(
            "close",
            class_name="person",
            min_bbox_height=200,
        )
        node.setup()
        assert node.update() == SUCCESS
