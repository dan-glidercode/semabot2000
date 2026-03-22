"""Integration tests for BehaviorTreeEngine with the full tree.

Five scenarios matching the spike results:
1. No detections          -> Wander (keys contain forward)
2. Person far left        -> Navigate + camera_left
3. Person far right       -> Navigate + camera_right
4. Person centered but far -> Navigate forward only
5. Person centered & close -> Interact (press E)
"""

from __future__ import annotations

import py_trees
import pytest

from semabot.core.config import (
    GameBehaviorConfig,
    GameControls,
    GameDetectionConfig,
    GameProfile,
)
from semabot.core.models import (
    BoundingBox,
    Detection,
    GameState,
)
from semabot.intelligence.behavior.engine import (
    BehaviorTreeEngine,
)
from semabot.intelligence.behavior.trees import (
    build_steal_a_brainrot_tree,
)

# ---------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    """Reset the global blackboard before each test."""
    py_trees.blackboard.Blackboard.enable_activity_stream()


@pytest.fixture()
def profile() -> GameProfile:
    return GameProfile(
        name="Steal a Brainrot",
        window_title="Roblox",
        controls=GameControls(
            move_forward="w",
            move_backward="s",
            move_left="a",
            move_right="d",
            camera_left="left",
            camera_right="right",
            jump="space",
            interact="e",
        ),
        detection=GameDetectionConfig(
            target_classes=("person",),
            target_min_confidence=0.4,
        ),
        behavior=GameBehaviorConfig(
            approach_tolerance_px=60,
            close_enough_bbox_height=200,
            wander_cycle_ticks=30,
        ),
    )


@pytest.fixture()
def engine(profile: GameProfile) -> BehaviorTreeEngine:
    tree = build_steal_a_brainrot_tree(profile)
    return BehaviorTreeEngine(tree)


# ---------------------------------------------------------------- helpers


def _state(
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
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    conf: float = 0.7,
) -> Detection:
    return Detection(
        class_name="person",
        confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# ---------------------------------------------------------------- scenarios


class TestBehaviorTreeEngine:
    """End-to-end scenarios through the full tree."""

    def test_no_detections_wander(
        self,
        engine: BehaviorTreeEngine,
    ) -> None:
        """Scenario 1: no detections -> Wander."""
        action = engine.decide(_state())
        assert "w" in action.keys_press
        assert "Wander" in action.description

    def test_person_far_left_navigate_left(
        self,
        engine: BehaviorTreeEngine,
    ) -> None:
        """Scenario 2: person on far left -> camera_left."""
        state = _state((_det(x1=50, y1=200, x2=120, y2=400),))
        action = engine.decide(state)
        assert "w" in action.keys_press
        assert "left" in action.keys_press
        assert "Navigate" in action.description

    def test_person_far_right_navigate_right(
        self,
        engine: BehaviorTreeEngine,
    ) -> None:
        """Scenario 3: person on far right -> camera_right."""
        state = _state((_det(x1=500, y1=200, x2=580, y2=400),))
        action = engine.decide(state)
        assert "w" in action.keys_press
        assert "right" in action.keys_press
        assert "Navigate" in action.description

    def test_person_centered_but_far_forward(
        self,
        engine: BehaviorTreeEngine,
    ) -> None:
        """Scenario 4: centered, small bbox -> forward only."""
        # bbox height = 100 (< 200 threshold)
        state = _state((_det(x1=290, y1=300, x2=350, y2=400),))
        action = engine.decide(state)
        assert action.keys_press == ("w",)
        assert "Navigate" in action.description

    def test_person_centered_and_close_interact(
        self,
        engine: BehaviorTreeEngine,
    ) -> None:
        """Scenario 5: centered + close -> Interact (E)."""
        # bbox height = 400 (>= 200), center_x = 320
        state = _state((_det(x1=280, y1=100, x2=360, y2=500),))
        action = engine.decide(state)
        assert action.keys_press == ("e",)
        assert "Interact" in action.description
