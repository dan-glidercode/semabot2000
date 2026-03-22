"""Null (dry-run) input controller.

Logs every action without sending any input to the operating system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semabot.core.models import Action

logger = logging.getLogger(__name__)


class NullInputController:
    """InputController that only logs — sends nothing to the OS.

    Implements the ``InputController`` protocol defined in
    ``semabot.core.protocols``.

    Useful for ``--dry-run`` mode and automated testing.
    """

    # -- InputController protocol ------------------------------------

    def execute(self, action: Action) -> None:
        """Log the action's keys and description."""
        logger.info(
            "NullInputController.execute: keys_press=%s, " "description='%s'",
            action.keys_press,
            action.description,
        )

    def release_all(self) -> None:
        """Log that all keys would be released."""
        logger.info("NullInputController.release_all: all keys released")
