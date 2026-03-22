"""BehaviorTreeEngine — adapts py_trees to the DecisionEngine protocol."""

from __future__ import annotations

import py_trees

from semabot.core.models import Action, GameState


class BehaviorTreeEngine:
    """Wraps a ``py_trees.trees.BehaviourTree`` as a DecisionEngine.

    Usage::

        tree = build_steal_a_brainrot_tree(profile)
        engine = BehaviorTreeEngine(tree)
        action = engine.decide(game_state)
    """

    def __init__(self, tree: py_trees.trees.BehaviourTree) -> None:
        self._tree = tree
        self._bb = py_trees.blackboard.Client(
            name="BehaviorTreeEngine",
        )
        self._bb.register_key(
            key="game_state",
            access=py_trees.common.Access.WRITE,
        )
        self._bb.register_key(
            key="action",
            access=py_trees.common.Access.READ,
        )

    def decide(self, state: GameState) -> Action:
        """Write *state* to the blackboard, tick, and return the Action.

        Returns a default empty ``Action`` when the tree does not
        produce one.
        """
        self._bb.game_state = state
        self._tree.tick()
        action: Action | None = self._bb.action
        if action is None or not isinstance(action, Action):
            return Action()
        return action
