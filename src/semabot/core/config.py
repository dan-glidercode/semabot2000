"""Configuration dataclasses and TOML loaders.

All config is loaded from TOML files via ``tomllib`` (stdlib 3.11+).
No external dependencies.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Bot configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureConfig:
    """Screen-capture settings."""

    method: str
    window_title: str


@dataclass(frozen=True)
class DetectionConfig:
    """YOLO / ONNX detection settings."""

    model_path: str
    provider: str
    confidence_threshold: float
    input_size: int
    nms_iou_threshold: float


@dataclass(frozen=True)
class DecisionConfig:
    """Behavior-tree tick rate."""

    tree_tick_rate_hz: int


@dataclass(frozen=True)
class ActionConfig:
    """Input-simulation settings."""

    method: str
    key_hold_duration_ms: int


@dataclass(frozen=True)
class BotConfig:
    """Top-level bot configuration (loaded from *default.toml*)."""

    target_fps: int
    log_level: str
    capture: CaptureConfig
    detection: DetectionConfig
    decision: DecisionConfig
    action: ActionConfig


# ---------------------------------------------------------------------------
# Game profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameControls:
    """Key bindings for a specific game."""

    move_forward: str
    move_backward: str
    move_left: str
    move_right: str
    camera_left: str
    camera_right: str
    jump: str
    interact: str


@dataclass(frozen=True)
class GameDetectionConfig:
    """Which classes / confidence the game cares about."""

    target_classes: tuple[str, ...]
    target_min_confidence: float


@dataclass(frozen=True)
class GameBehaviorConfig:
    """Tuning knobs for the behavior tree."""

    approach_tolerance_px: int
    close_enough_bbox_height: int
    wander_cycle_ticks: int


@dataclass(frozen=True)
class GameProfile:
    """Per-game configuration (loaded from *config/games/<name>.toml*)."""

    name: str
    window_title: str
    controls: GameControls
    detection: GameDetectionConfig
    behavior: GameBehaviorConfig


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> BotConfig:
    """Read a TOML file at *path* and return a ``BotConfig``.

    Raises ``KeyError`` when required keys are missing and
    ``FileNotFoundError`` when *path* does not exist.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    bot = raw["bot"]
    cap = raw["capture"]
    det = raw["detection"]
    dec = raw["decision"]
    act = raw["action"]

    return BotConfig(
        target_fps=bot["target_fps"],
        log_level=bot["log_level"],
        capture=CaptureConfig(
            method=cap["method"],
            window_title=cap["window_title"],
        ),
        detection=DetectionConfig(
            model_path=det["model_path"],
            provider=det["provider"],
            confidence_threshold=det["confidence_threshold"],
            input_size=det["input_size"],
            nms_iou_threshold=det["nms_iou_threshold"],
        ),
        decision=DecisionConfig(
            tree_tick_rate_hz=dec["tree_tick_rate_hz"],
        ),
        action=ActionConfig(
            method=act["method"],
            key_hold_duration_ms=act["key_hold_duration_ms"],
        ),
    )


def load_game_profile(path: str | Path) -> GameProfile:
    """Read a TOML file at *path* and return a ``GameProfile``.

    Raises ``KeyError`` when required keys are missing and
    ``FileNotFoundError`` when *path* does not exist.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    game = raw["game"]
    ctrl = raw["controls"]
    det = raw["detection"]
    beh = raw["behavior"]

    return GameProfile(
        name=game["name"],
        window_title=game["window_title"],
        controls=GameControls(
            move_forward=ctrl["move_forward"],
            move_backward=ctrl["move_backward"],
            move_left=ctrl["move_left"],
            move_right=ctrl["move_right"],
            camera_left=ctrl["camera_left"],
            camera_right=ctrl["camera_right"],
            jump=ctrl["jump"],
            interact=ctrl["interact"],
        ),
        detection=GameDetectionConfig(
            target_classes=tuple(det["target_classes"]),
            target_min_confidence=det["target_min_confidence"],
        ),
        behavior=GameBehaviorConfig(
            approach_tolerance_px=beh["approach_tolerance_px"],
            close_enough_bbox_height=beh["close_enough_bbox_height"],
            wander_cycle_ticks=beh["wander_cycle_ticks"],
        ),
    )
