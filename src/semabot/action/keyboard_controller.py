"""Keyboard input controller using pydirectinput.

Translates :class:`~semabot.core.models.Action` objects into real
key presses / releases via the Windows ``SendInput`` API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semabot.core.models import Action

logger = logging.getLogger(__name__)


class KeyboardController:
    """InputController backed by pydirectinput.

    Implements the ``InputController`` protocol defined in
    ``semabot.core.protocols``.

    Tracks which keys are currently held so that redundant
    ``keyDown`` / ``keyUp`` calls are avoided.
    """

    def __init__(self) -> None:
        self._held: set[str] = set()
        self._pdi = self._init_pydirectinput()

    # -- InputController protocol ------------------------------------

    def execute(self, action: Action) -> None:
        """Diff *action.keys_press* against held keys.

        * Keys in ``keys_press`` but not currently held are pressed.
        * Keys currently held but absent from ``keys_press`` are
          released.
        """
        desired = set(action.keys_press)
        to_release = self._held - desired
        to_press = desired - self._held

        for key in to_release:
            self._pdi.keyUp(key)
            logger.debug("keyUp: %s", key)

        for key in to_press:
            self._pdi.keyDown(key)
            logger.debug("keyDown: %s", key)

        self._held = desired

    def release_all(self) -> None:
        """Release every key that is currently held."""
        for key in self._held:
            self._pdi.keyUp(key)
            logger.debug("keyUp (release_all): %s", key)
        self._held.clear()

    # -- internals ---------------------------------------------------

    @staticmethod
    def _init_pydirectinput():  # noqa: ANN205
        """Lazy-import pydirectinput and configure it."""
        import pydirectinput  # noqa: WPS433

        pydirectinput.PAUSE = 0
        return pydirectinput
