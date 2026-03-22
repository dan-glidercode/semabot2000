"""Tests for behavior-tree action nodes."""

from __future__ import annotations

import py_trees
import pytest

from semabot.core.models import (
    Action,
    BoundingBox,
    Detection,
    GameState,
)
from semabot.intelligence.behavior.actions import (
    InteractAction,
    NavigateToTarget,
    WanderAction,
)

SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE
RUNNING = py_trees.common.Status.RUNNING


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


def _read_action() -> Action:
    bb = py_trees.blackboard.Client(name="test_reader")
    bb.register_key(
        key="action",
        access=py_trees.common.Access.READ,
    )
    return bb.action


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    py_trees.blackboard.Blackboard.enable_activity_stream()
    _setup_blackboard(_make_state())


# --------------------------------------------------------- NavigateToTarget


class TestNavigateToTarget:
    """Tests for NavigateToTarget action node."""

    def _make_node(
        self,
        tolerance: float = 60,
    ) -> NavigateToTarget:
        return NavigateToTarget(
            "nav",
            class_name="person",
            forward_key="w",
            camera_left_key="left",
            camera_right_key="right",
            tolerance_px=tolerance,
        )

    def test_target_left_adds_camera_left(self) -> None:
        # center_x of det = 50, frame center = 320
        state = _make_state((_det(x1=0, y1=0, x2=100, y2=100),))
        _setup_blackboard(state)
        node = self._make_node()
        node.setup()
        status = node.update()
        assert status == RUNNING
        action = _read_action()
        assert "w" in action.keys_press
        assert "left" in action.keys_press

    def test_target_right_adds_camera_right(self) -> None:
        # center_x of det = 590, frame center = 320
        state = _make_state((_det(x1=580, y1=0, x2=600, y2=100),))
        _setup_blackboard(state)
        node = self._make_node()
        node.setup()
        status = node.update()
        assert status == RUNNING
        action = _read_action()
        assert "w" in action.keys_press
        assert "right" in action.keys_press

    def test_target_center_forward_only(self) -> None:
        # center_x of det = 320 (frame center)
        state = _make_state((_det(x1=300, y1=0, x2=340, y2=100),))
        _setup_blackboard(state)
        node = self._make_node()
        node.setup()
        status = node.update()
        assert status == RUNNING
        action = _read_action()
        assert action.keys_press == ("w",)

    def test_no_target_returns_failure(self) -> None:
        _setup_blackboard(_make_state())
        node = self._make_node()
        node.setup()
        assert node.update() == FAILURE

    def test_description_contains_class_name(self) -> None:
        state = _make_state((_det(x1=300, y1=0, x2=340, y2=100),))
        _setup_blackboard(state)
        node = self._make_node()
        node.setup()
        node.update()
        action = _read_action()
        assert "person" in action.description


# -------------------------------------------------------------- InteractAction


class TestInteractAction:
    """Tests for InteractAction node."""

    def test_correct_key_in_action(self) -> None:
        node = InteractAction("interact", key="e")
        node.setup()
        status = node.update()
        assert status == SUCCESS
        action = _read_action()
        assert action.keys_press == ("e",)

    def test_description_contains_key(self) -> None:
        node = InteractAction("interact", key="e")
        node.setup()
        node.update()
        action = _read_action()
        assert "e" in action.description

    def test_different_key(self) -> None:
        node = InteractAction("interact", key="f")
        node.setup()
        node.update()
        action = _read_action()
        assert action.keys_press == ("f",)


# -------------------------------------------------------------- WanderAction


class TestWanderAction:
    """Tests for WanderAction node."""

    def _make_node(
        self,
        cycle_ticks: int = 4,
    ) -> WanderAction:
        return WanderAction(
            "wander",
            forward_key="w",
            camera_left_key="left",
            camera_right_key="right",
            cycle_ticks=cycle_ticks,
        )

    def test_always_returns_running(self) -> None:
        node = self._make_node()
        node.setup()
        for _ in range(12):
            assert node.update() == RUNNING

    def test_always_includes_forward_key(self) -> None:
        node = self._make_node()
        node.setup()
        for _ in range(12):
            node.update()
            action = _read_action()
            assert "w" in action.keys_press

    def test_cycles_through_directions(self) -> None:
        """Verify the wander pattern: left, fwd, right, fwd."""
        node = self._make_node(cycle_ticks=2)
        node.setup()

        directions: list[str] = []
        for _ in range(8):
            node.update()
            action = _read_action()
            if "left" in action.keys_press:
                directions.append("left")
            elif "right" in action.keys_press:
                directions.append("right")
            else:
                directions.append("fwd")

        # tick_count increments before computing the cycle:
        #   1//2=0 left, 2//2=1 fwd, 3//2=1 fwd, 4//2=2 right,
        #   5//2=2 right, 6//2=3 fwd, 7//2=3 fwd, 8//2=0 left
        assert directions == [
            "left",
            "fwd",
            "fwd",
            "right",
            "right",
            "fwd",
            "fwd",
            "left",
        ]
