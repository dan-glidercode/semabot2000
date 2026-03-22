"""Tests for the component factory."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semabot.app.factory import create_bot
from semabot.app.orchestrator import BotOrchestrator

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


_DEFAULT_TOML = textwrap.dedent("""\
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
""")

_GAME_TOML = textwrap.dedent("""\
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
""")


@pytest.fixture()
def config_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write TOML files and return (config_path, profile_path)."""
    config_path = tmp_path / "default.toml"
    config_path.write_text(_DEFAULT_TOML)

    profile_path = tmp_path / "steal_a_brainrot.toml"
    profile_path.write_text(_GAME_TOML)

    return config_path, profile_path


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestCreateBot:
    """Factory produces a correctly-wired BotOrchestrator."""

    @patch(
        "semabot.capture.wgc_source.WGCFrameSource",
        autospec=True,
    )
    def test_creates_orchestrator(
        self,
        mock_wgc: MagicMock,
        config_files: tuple[Path, Path],
    ) -> None:
        config_path, profile_path = config_files
        bot = create_bot(config_path, profile_path)

        assert isinstance(bot, BotOrchestrator)

    @patch(
        "semabot.capture.wgc_source.WGCFrameSource",
        autospec=True,
    )
    def test_dry_run_uses_null_controller(
        self,
        mock_wgc: MagicMock,
        config_files: tuple[Path, Path],
    ) -> None:
        config_path, profile_path = config_files
        bot = create_bot(config_path, profile_path, dry_run=True)

        from semabot.action.null_controller import (
            NullInputController,
        )

        # Access private attribute for verification
        ctrl = bot._input_controller  # noqa: WPS437
        assert isinstance(ctrl, NullInputController)

    @patch(
        "semabot.action.keyboard_controller.pydirectinput",
        create=True,
    )
    @patch(
        "semabot.capture.wgc_source.WGCFrameSource",
        autospec=True,
    )
    def test_no_dry_run_uses_keyboard_controller(
        self,
        mock_wgc: MagicMock,
        mock_pdi: MagicMock,
        config_files: tuple[Path, Path],
    ) -> None:
        config_path, profile_path = config_files

        with patch(
            "semabot.action.keyboard_controller" ".KeyboardController._init_pydirectinput",
            return_value=mock_pdi,
        ):
            bot = create_bot(
                config_path,
                profile_path,
                dry_run=False,
            )

        from semabot.action.keyboard_controller import (
            KeyboardController,
        )

        ctrl = bot._input_controller  # noqa: WPS437
        assert isinstance(ctrl, KeyboardController)


class TestCaptureMethodSelection:
    """Factory picks the right frame source based on config."""

    @patch(
        "semabot.capture.wgc_source.WGCFrameSource",
        autospec=True,
    )
    def test_wgc_method(
        self,
        mock_wgc: MagicMock,
        config_files: tuple[Path, Path],
    ) -> None:
        config_path, profile_path = config_files
        create_bot(config_path, profile_path, dry_run=True)

        mock_wgc.assert_called_once_with(window_title="Roblox")

    def test_mss_method(
        self,
        config_files: tuple[Path, Path],
    ) -> None:
        config_path, profile_path = config_files
        # Rewrite config with mss method
        mss_toml = _DEFAULT_TOML.replace('method = "wgc"', 'method = "mss"')
        config_path.write_text(mss_toml)

        with patch(
            "semabot.capture.mss_source.MSSFrameSource",
            autospec=True,
        ) as mock_mss:
            create_bot(config_path, profile_path, dry_run=True)
            mock_mss.assert_called_once()
