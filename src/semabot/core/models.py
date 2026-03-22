"""Core domain models — pure data, no behavior, no external dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        """Return the (cx, cy) center point of the box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        """Return the width of the box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return the height of the box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Return the area of the box."""
        return self.width * self.height


@dataclass(frozen=True)
class Detection:
    """A single detected object in a frame."""

    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of the game world at a point in time."""

    detections: tuple[Detection, ...]
    frame_width: int
    frame_height: int
    timestamp: float


@dataclass(frozen=True)
class Action:
    """Describes what keys to press/release this frame."""

    keys_press: tuple[str, ...] = ()
    keys_release: tuple[str, ...] = ()
    description: str = ""
