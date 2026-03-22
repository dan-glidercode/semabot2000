"""Abstract interfaces (protocols) for the SeMaBot2000 pipeline.

No concrete implementations live here — only contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from semabot.core.models import Action, Detection, GameState

if TYPE_CHECKING:
    import numpy as np


class FrameSource(Protocol):
    """Provides raw frames from the game."""

    def start(self) -> None:
        """Begin capturing frames."""
        ...

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recent frame, or None if unavailable."""
        ...

    def stop(self) -> None:
        """Stop capturing and release resources."""
        ...


class Preprocessor(Protocol):
    """Transforms a raw frame into model input."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and reshape *frame* for the detector."""
        ...


class Detector(Protocol):
    """Runs object detection on a preprocessed frame."""

    def detect(self, blob: np.ndarray) -> list[Detection]:
        """Return detections found in *blob*."""
        ...


class StateBuilder(Protocol):
    """Builds game state from detections and frame metadata."""

    def build(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> GameState:
        """Combine *detections* and *frame* metadata into a GameState."""
        ...


class DecisionEngine(Protocol):
    """Decides what action to take given the current game state."""

    def decide(self, state: GameState) -> Action:
        """Return the Action to execute for the given *state*."""
        ...


class InputController(Protocol):
    """Sends actions to the game."""

    def execute(self, action: Action) -> None:
        """Press/release keys described by *action*."""
        ...

    def release_all(self) -> None:
        """Release every key that is currently held."""
        ...
