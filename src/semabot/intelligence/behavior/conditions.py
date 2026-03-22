"""Condition nodes for the behavior tree.

Each node reads ``game_state`` from the py_trees blackboard and
returns SUCCESS or FAILURE based on the current detections.
"""

from __future__ import annotations

import py_trees

from semabot.core.models import Detection, GameState


def _best_match(
    state: GameState,
    class_name: str,
    min_confidence: float = 0.0,
) -> Detection | None:
    """Return the highest-confidence detection for *class_name*."""
    best: Detection | None = None
    for det in state.detections:
        if det.class_name != class_name:
            continue
        if det.confidence < min_confidence:
            continue
        if best is None or det.confidence > best.confidence:
            best = det
    return best


class HasDetection(py_trees.behaviour.Behaviour):
    """SUCCESS when any detection matches *class_name* and confidence.

    Reads the ``game_state`` key from the blackboard.
    """

    def __init__(
        self,
        name: str,
        class_name: str,
        min_confidence: float = 0.3,
    ) -> None:
        super().__init__(name)
        self.class_name = class_name
        self.min_confidence = min_confidence
        self._bb = self.attach_blackboard_client()
        self._bb.register_key(
            key="game_state",
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        state: GameState = self._bb.game_state
        match = _best_match(
            state,
            self.class_name,
            self.min_confidence,
        )
        if match is not None:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class TargetInCenter(py_trees.behaviour.Behaviour):
    """SUCCESS when the best detection's center_x is near frame center.

    "Near" is defined by *tolerance_px* (pixels either side).
    Reads the ``game_state`` key from the blackboard.
    """

    def __init__(
        self,
        name: str,
        class_name: str,
        tolerance_px: float = 60,
    ) -> None:
        super().__init__(name)
        self.class_name = class_name
        self.tolerance_px = tolerance_px
        self._bb = self.attach_blackboard_client()
        self._bb.register_key(
            key="game_state",
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        state: GameState = self._bb.game_state
        best = _best_match(state, self.class_name)
        if best is None:
            return py_trees.common.Status.FAILURE
        frame_center_x = state.frame_width / 2
        det_center_x = best.bbox.center[0]
        if abs(det_center_x - frame_center_x) <= self.tolerance_px:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class TargetClose(py_trees.behaviour.Behaviour):
    """SUCCESS when the best detection's bbox height >= threshold.

    Reads the ``game_state`` key from the blackboard.
    """

    def __init__(
        self,
        name: str,
        class_name: str,
        min_bbox_height: float = 200,
    ) -> None:
        super().__init__(name)
        self.class_name = class_name
        self.min_bbox_height = min_bbox_height
        self._bb = self.attach_blackboard_client()
        self._bb.register_key(
            key="game_state",
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        state: GameState = self._bb.game_state
        best = _best_match(state, self.class_name)
        if best is None:
            return py_trees.common.Status.FAILURE
        if best.bbox.height >= self.min_bbox_height:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
