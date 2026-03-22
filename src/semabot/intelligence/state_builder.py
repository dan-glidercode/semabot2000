"""Game state builder — assemble detections into an immutable snapshot."""

from __future__ import annotations

import time

import numpy as np

from semabot.core.models import Detection, GameState


class GameStateBuilder:
    """Implements the StateBuilder protocol.

    Filters detections below *confidence_threshold*, attaches frame
    dimensions and a timestamp, and returns a frozen
    :class:`GameState`.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
    ) -> None:
        self._confidence_threshold = confidence_threshold

    def build(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> GameState:
        """Combine *detections* and *frame* metadata into a GameState.

        Parameters
        ----------
        detections:
            Raw detections from the detector.
        frame:
            The original BGR frame (used only for shape).

        Returns
        -------
        GameState
            Frozen dataclass with filtered detections, frame
            dimensions, and current timestamp.
        """
        filtered = tuple(d for d in detections if d.confidence >= self._confidence_threshold)

        frame_height, frame_width = frame.shape[:2]

        return GameState(
            detections=filtered,
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            timestamp=time.time(),
        )
