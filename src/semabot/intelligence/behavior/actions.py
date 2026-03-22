"""Action nodes for the behavior tree.

Each node writes an ``Action`` to the ``action`` key on the py_trees
blackboard so that downstream consumers (e.g. BehaviorTreeEngine) can
read the chosen action after a tick.
"""

from __future__ import annotations

import py_trees

from semabot.core.models import Action, Detection, GameState


def _best_match(
    state: GameState,
    class_name: str,
) -> Detection | None:
    """Return the highest-confidence detection for *class_name*."""
    best: Detection | None = None
    for det in state.detections:
        if det.class_name != class_name:
            continue
        if best is None or det.confidence > best.confidence:
            best = det
    return best


class NavigateToTarget(py_trees.behaviour.Behaviour):
    """Move toward the best detection of *class_name*.

    Writes an ``Action`` containing *forward_key* plus a camera
    rotation key when the target is offset from frame center.
    Returns RUNNING while navigating, FAILURE if no target found.
    """

    def __init__(
        self,
        name: str,
        class_name: str,
        forward_key: str,
        camera_left_key: str,
        camera_right_key: str,
        tolerance_px: float = 60,
    ) -> None:
        super().__init__(name)
        self.class_name = class_name
        self.forward_key = forward_key
        self.camera_left_key = camera_left_key
        self.camera_right_key = camera_right_key
        self.tolerance_px = tolerance_px
        self._bb = self.attach_blackboard_client()
        self._bb.register_key(
            key="game_state",
            access=py_trees.common.Access.READ,
        )
        self._bb.register_key(
            key="action",
            access=py_trees.common.Access.WRITE,
        )

    def update(self) -> py_trees.common.Status:
        state: GameState = self._bb.game_state
        best = _best_match(state, self.class_name)
        if best is None:
            return py_trees.common.Status.FAILURE

        keys: list[str] = [self.forward_key]
        frame_center_x = state.frame_width / 2
        offset_x = best.bbox.center[0] - frame_center_x

        if offset_x < -self.tolerance_px:
            keys.append(self.camera_left_key)
        elif offset_x > self.tolerance_px:
            keys.append(self.camera_right_key)

        cx, cy = best.bbox.center
        self._bb.action = Action(
            keys_press=tuple(keys),
            description=(f"Navigate to {self.class_name}" f" at ({cx:.0f}, {cy:.0f})"),
        )
        return py_trees.common.Status.RUNNING


class InteractAction(py_trees.behaviour.Behaviour):
    """Press a single interaction key and return SUCCESS."""

    def __init__(self, name: str, key: str) -> None:
        super().__init__(name)
        self.key = key
        self._bb = self.attach_blackboard_client()
        self._bb.register_key(
            key="action",
            access=py_trees.common.Access.WRITE,
        )

    def update(self) -> py_trees.common.Status:
        self._bb.action = Action(
            keys_press=(self.key,),
            description=f"Interact ({self.key})",
        )
        return py_trees.common.Status.SUCCESS


class WanderAction(py_trees.behaviour.Behaviour):
    """Walk forward with cyclic camera rotation.

    The rotation direction changes every *cycle_ticks* ticks:
      cycle 0 -> camera left
      cycle 1 -> forward only
      cycle 2 -> camera right
      cycle 3 -> forward only
    Always returns RUNNING.
    """

    def __init__(
        self,
        name: str,
        forward_key: str,
        camera_left_key: str,
        camera_right_key: str,
        cycle_ticks: int = 30,
    ) -> None:
        super().__init__(name)
        self.forward_key = forward_key
        self.camera_left_key = camera_left_key
        self.camera_right_key = camera_right_key
        self.cycle_ticks = cycle_ticks
        self._tick_count = 0
        self._bb = self.attach_blackboard_client()
        self._bb.register_key(
            key="action",
            access=py_trees.common.Access.WRITE,
        )

    def update(self) -> py_trees.common.Status:
        self._tick_count += 1
        cycle = (self._tick_count // self.cycle_ticks) % 4

        keys: list[str] = [self.forward_key]
        if cycle == 0:
            keys.append(self.camera_left_key)
        elif cycle == 2:
            keys.append(self.camera_right_key)

        self._bb.action = Action(
            keys_press=tuple(keys),
            description=f"Wander (cycle={cycle})",
        )
        return py_trees.common.Status.RUNNING
