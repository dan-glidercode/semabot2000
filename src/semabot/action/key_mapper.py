"""Maps logical control names to configured key bindings.

Uses a :class:`~semabot.core.config.GameControls` instance to resolve
human-readable names like ``"move_forward"`` into actual key strings.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semabot.core.config import GameControls

# All field names on GameControls that are considered required.
_REQUIRED_CONTROLS: frozenset[str] = frozenset(
    "move_forward move_backward move_left move_right "
    "camera_left camera_right jump interact".split()
)


class KeyMapper:
    """Translates control names to their configured key bindings.

    Parameters
    ----------
    controls:
        A :class:`GameControls` dataclass with key bindings.
    """

    def __init__(self, controls: GameControls) -> None:
        self._controls = controls
        # Build a dict from field-name -> value for fast lookup.
        self._map: dict[str, str] = {f.name: getattr(controls, f.name) for f in fields(controls)}

    def get_key(self, control_name: str) -> str:
        """Return the key bound to *control_name*.

        Raises ``KeyError`` if the name is unknown.
        """
        try:
            return self._map[control_name]
        except KeyError:
            raise KeyError(f"Unknown control name: '{control_name}'") from None

    def get_keys(self, control_names: list[str]) -> list[str]:
        """Return keys for each name in *control_names*."""
        return [self.get_key(name) for name in control_names]

    def validate(self) -> None:
        """Check that all required controls exist.

        Raises ``ValueError`` listing any missing controls.
        """
        present = set(self._map)
        missing = _REQUIRED_CONTROLS - present
        if missing:
            raise ValueError("Missing required controls: " + ", ".join(sorted(missing)))
