"""Pre-built behavior trees for supported games."""

from __future__ import annotations

import py_trees

from semabot.core.config import GameProfile
from semabot.core.models import Action, GameState
from semabot.intelligence.behavior.actions import (
    InteractAction,
    NavigateToTarget,
    WanderAction,
)
from semabot.intelligence.behavior.conditions import (
    HasDetection,
    TargetClose,
    TargetInCenter,
)


def build_steal_a_brainrot_tree(
    profile: GameProfile,
) -> py_trees.trees.BehaviourTree:
    """Build the *Steal a Brainrot* behavior tree.

    Tree layout::

        Selector (Root)
        +-- Sequence (Steal)
        |   +-- HasDetection
        |   +-- TargetInCenter
        |   +-- TargetClose
        |   +-- InteractAction
        +-- Sequence (Approach)
        |   +-- HasDetection
        |   +-- NavigateToTarget
        +-- WanderAction
    """
    ctrl = profile.controls
    det = profile.detection
    beh = profile.behavior
    target = det.target_classes[0]

    # --- Branch 1: Steal (close + centered) -------------------------
    steal = py_trees.composites.Sequence(
        "Steal",
        memory=False,
    )
    steal.add_children(
        [
            HasDetection(
                "HasDetection?",
                class_name=target,
                min_confidence=det.target_min_confidence,
            ),
            TargetInCenter(
                "Centered?",
                class_name=target,
                tolerance_px=float(beh.approach_tolerance_px),
            ),
            TargetClose(
                "Close?",
                class_name=target,
                min_bbox_height=float(beh.close_enough_bbox_height),
            ),
            InteractAction("Interact", key=ctrl.interact),
        ]
    )

    # --- Branch 2: Approach -----------------------------------------
    approach = py_trees.composites.Sequence(
        "Approach",
        memory=False,
    )
    approach.add_children(
        [
            HasDetection(
                "HasDetection?",
                class_name=target,
                min_confidence=det.target_min_confidence,
            ),
            NavigateToTarget(
                "Navigate",
                class_name=target,
                forward_key=ctrl.move_forward,
                camera_left_key=ctrl.camera_left,
                camera_right_key=ctrl.camera_right,
                tolerance_px=float(beh.approach_tolerance_px),
            ),
        ]
    )

    # --- Branch 3: Wander -------------------------------------------
    wander = WanderAction(
        "Wander",
        forward_key=ctrl.move_forward,
        camera_left_key=ctrl.camera_left,
        camera_right_key=ctrl.camera_right,
        cycle_ticks=beh.wander_cycle_ticks,
    )

    # --- Root selector ----------------------------------------------
    root = py_trees.composites.Selector(
        "Root",
        memory=False,
    )
    root.add_children([steal, approach, wander])

    tree = py_trees.trees.BehaviourTree(root=root)

    # Initialise blackboard keys so readers never see missing data.
    bb = py_trees.blackboard.Client(name="tree_setup")
    bb.register_key(
        key="game_state",
        access=py_trees.common.Access.WRITE,
    )
    bb.register_key(
        key="action",
        access=py_trees.common.Access.WRITE,
    )
    bb.game_state = GameState(
        detections=(),
        frame_width=640,
        frame_height=640,
        timestamp=0.0,
    )
    bb.action = Action()

    return tree
