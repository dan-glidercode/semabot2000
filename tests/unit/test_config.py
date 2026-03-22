"""Tests for configuration dataclasses and TOML loaders."""

from __future__ import annotations

import pytest

from semabot.core.config import (
    ActionConfig,
    BotConfig,
    CaptureConfig,
    DecisionConfig,
    DetectionConfig,
    GameBehaviorConfig,
    GameControls,
    GameDetectionConfig,
    GameProfile,
    load_config,
    load_game_profile,
)

# ---------------------------------------------------------------------------
# Sample TOML content
# ---------------------------------------------------------------------------

VALID_BOT_TOML = """\
[bot]
target_fps = 25
log_level = "INFO"

[capture]
method = "wgc"
window_title = "Roblox"

[detection]
model_path = "models/yolo11n.onnx"
provider = "DmlExecutionProvider"
confidence_threshold = 0.35
input_size = 640
nms_iou_threshold = 0.5

[decision]
tree_tick_rate_hz = 25

[action]
method = "pydirectinput"
key_hold_duration_ms = 50
"""

VALID_GAME_TOML = """\
[game]
name = "Steal a Brainrot"
window_title = "Roblox"

[controls]
move_forward = "w"
move_backward = "s"
move_left = "a"
move_right = "d"
camera_left = "left"
camera_right = "right"
jump = "space"
interact = "e"

[detection]
target_classes = ["person"]
target_min_confidence = 0.4

[behavior]
approach_tolerance_px = 60
close_enough_bbox_height = 200
wander_cycle_ticks = 30
"""


# ---------------------------------------------------------------------------
# Dataclass construction tests
# ---------------------------------------------------------------------------


class TestCaptureConfig:
    def test_construction(self) -> None:
        cfg = CaptureConfig(method="wgc", window_title="Roblox")
        assert cfg.method == "wgc"
        assert cfg.window_title == "Roblox"


class TestDetectionConfig:
    def test_construction(self) -> None:
        cfg = DetectionConfig(
            model_path="models/yolo11n.onnx",
            provider="DmlExecutionProvider",
            confidence_threshold=0.35,
            input_size=640,
            nms_iou_threshold=0.5,
        )
        assert cfg.model_path == "models/yolo11n.onnx"
        assert cfg.confidence_threshold == 0.35
        assert cfg.input_size == 640


class TestDecisionConfig:
    def test_construction(self) -> None:
        cfg = DecisionConfig(tree_tick_rate_hz=25)
        assert cfg.tree_tick_rate_hz == 25


class TestActionConfig:
    def test_construction(self) -> None:
        cfg = ActionConfig(method="pydirectinput", key_hold_duration_ms=50)
        assert cfg.method == "pydirectinput"
        assert cfg.key_hold_duration_ms == 50


class TestBotConfig:
    def test_construction(self) -> None:
        cfg = BotConfig(
            target_fps=25,
            log_level="INFO",
            capture=CaptureConfig(method="wgc", window_title="Roblox"),
            detection=DetectionConfig(
                model_path="m.onnx",
                provider="Dml",
                confidence_threshold=0.3,
                input_size=640,
                nms_iou_threshold=0.5,
            ),
            decision=DecisionConfig(tree_tick_rate_hz=25),
            action=ActionConfig(
                method="pydirectinput",
                key_hold_duration_ms=50,
            ),
        )
        assert cfg.target_fps == 25
        assert cfg.capture.method == "wgc"
        assert cfg.detection.model_path == "m.onnx"


class TestGameControls:
    def test_construction(self) -> None:
        ctrl = GameControls(
            move_forward="w",
            move_backward="s",
            move_left="a",
            move_right="d",
            camera_left="left",
            camera_right="right",
            jump="space",
            interact="e",
        )
        assert ctrl.move_forward == "w"
        assert ctrl.jump == "space"


class TestGameDetectionConfig:
    def test_construction(self) -> None:
        cfg = GameDetectionConfig(
            target_classes=("person",),
            target_min_confidence=0.4,
        )
        assert cfg.target_classes == ("person",)
        assert cfg.target_min_confidence == 0.4


class TestGameBehaviorConfig:
    def test_construction(self) -> None:
        cfg = GameBehaviorConfig(
            approach_tolerance_px=60,
            close_enough_bbox_height=200,
            wander_cycle_ticks=30,
        )
        assert cfg.approach_tolerance_px == 60


class TestGameProfile:
    def test_construction(self) -> None:
        profile = GameProfile(
            name="Test Game",
            window_title="Roblox",
            controls=GameControls(
                move_forward="w",
                move_backward="s",
                move_left="a",
                move_right="d",
                camera_left="left",
                camera_right="right",
                jump="space",
                interact="e",
            ),
            detection=GameDetectionConfig(
                target_classes=("person",),
                target_min_confidence=0.4,
            ),
            behavior=GameBehaviorConfig(
                approach_tolerance_px=60,
                close_enough_bbox_height=200,
                wander_cycle_ticks=30,
            ),
        )
        assert profile.name == "Test Game"
        assert profile.controls.interact == "e"


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_toml(self, tmp_path) -> None:
        toml_file = tmp_path / "bot.toml"
        toml_file.write_text(VALID_BOT_TOML, encoding="utf-8")
        cfg = load_config(toml_file)

        assert isinstance(cfg, BotConfig)
        assert cfg.target_fps == 25
        assert cfg.log_level == "INFO"
        assert cfg.capture.method == "wgc"
        assert cfg.capture.window_title == "Roblox"
        assert cfg.detection.model_path == "models/yolo11n.onnx"
        assert cfg.detection.provider == "DmlExecutionProvider"
        assert cfg.detection.confidence_threshold == 0.35
        assert cfg.detection.input_size == 640
        assert cfg.detection.nms_iou_threshold == 0.5
        assert cfg.decision.tree_tick_rate_hz == 25
        assert cfg.action.method == "pydirectinput"
        assert cfg.action.key_hold_duration_ms == 50

    def test_missing_section_raises_key_error(self, tmp_path) -> None:
        incomplete = '[bot]\ntarget_fps = 25\nlog_level = "INFO"\n'
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text(incomplete, encoding="utf-8")

        with pytest.raises(KeyError):
            load_config(toml_file)

    def test_missing_key_raises_key_error(self, tmp_path) -> None:
        # Missing 'model_path' in [detection]
        bad_toml = """\
[bot]
target_fps = 25
log_level = "INFO"

[capture]
method = "wgc"
window_title = "Roblox"

[detection]
provider = "DmlExecutionProvider"
confidence_threshold = 0.35
input_size = 640
nms_iou_threshold = 0.5

[decision]
tree_tick_rate_hz = 25

[action]
method = "pydirectinput"
key_hold_duration_ms = 50
"""
        toml_file = tmp_path / "bad2.toml"
        toml_file.write_text(bad_toml, encoding="utf-8")

        with pytest.raises(KeyError):
            load_config(toml_file)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/bot.toml")


# ---------------------------------------------------------------------------
# load_game_profile tests
# ---------------------------------------------------------------------------


class TestLoadGameProfile:
    def test_valid_toml(self, tmp_path) -> None:
        toml_file = tmp_path / "game.toml"
        toml_file.write_text(VALID_GAME_TOML, encoding="utf-8")
        profile = load_game_profile(toml_file)

        assert isinstance(profile, GameProfile)
        assert profile.name == "Steal a Brainrot"
        assert profile.window_title == "Roblox"
        assert profile.controls.move_forward == "w"
        assert profile.controls.jump == "space"
        assert profile.controls.interact == "e"
        assert profile.detection.target_classes == ("person",)
        assert profile.detection.target_min_confidence == 0.4
        assert profile.behavior.approach_tolerance_px == 60
        assert profile.behavior.close_enough_bbox_height == 200
        assert profile.behavior.wander_cycle_ticks == 30

    def test_missing_section_raises_key_error(self, tmp_path) -> None:
        incomplete = '[game]\nname = "Test"\nwindow_title = "Roblox"\n'
        toml_file = tmp_path / "bad_game.toml"
        toml_file.write_text(incomplete, encoding="utf-8")

        with pytest.raises(KeyError):
            load_game_profile(toml_file)

    def test_missing_key_raises_key_error(self, tmp_path) -> None:
        # Missing 'interact' in [controls]
        bad_toml = """\
[game]
name = "Bad"
window_title = "Roblox"

[controls]
move_forward = "w"
move_backward = "s"
move_left = "a"
move_right = "d"
camera_left = "left"
camera_right = "right"
jump = "space"

[detection]
target_classes = ["person"]
target_min_confidence = 0.4

[behavior]
approach_tolerance_px = 60
close_enough_bbox_height = 200
wander_cycle_ticks = 30
"""
        toml_file = tmp_path / "bad_game2.toml"
        toml_file.write_text(bad_toml, encoding="utf-8")

        with pytest.raises(KeyError):
            load_game_profile(toml_file)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_game_profile("/nonexistent/path/game.toml")
