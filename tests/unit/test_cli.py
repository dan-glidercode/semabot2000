"""Tests for CLI argument parsing and subcommand handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from semabot.app.cli import main, parse_args
from semabot.core.constants import DEFAULT_CONFIG_PATH

# -------------------------------------------------------------------
# ``run`` subcommand
# -------------------------------------------------------------------


class TestRunParser:
    """Argument parsing for the ``run`` subcommand."""

    def test_run_requires_game(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["run"])

    def test_run_game_flag(self) -> None:
        args = parse_args(["run", "--game", "steal_a_brainrot"])
        assert args.command == "run"
        assert args.game == "steal_a_brainrot"

    def test_run_defaults(self) -> None:
        args = parse_args(["run", "--game", "test_game"])
        assert args.config == DEFAULT_CONFIG_PATH
        assert args.dry_run is False
        assert args.save_detections is False
        assert args.duration is None
        assert args.log_level is None

    def test_run_dry_run_flag(self) -> None:
        args = parse_args(["run", "--game", "g", "--dry-run"])
        assert args.dry_run is True

    def test_run_duration(self) -> None:
        args = parse_args(["run", "--game", "g", "--duration", "30.5"])
        assert args.duration == 30.5

    def test_run_log_level(self) -> None:
        args = parse_args(["run", "--game", "g", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_run_save_detections(self) -> None:
        args = parse_args(["run", "--game", "g", "--save-detections"])
        assert args.save_detections is True

    def test_run_custom_config(self) -> None:
        args = parse_args(["run", "--game", "g", "--config", "my.toml"])
        assert args.config == "my.toml"


# -------------------------------------------------------------------
# ``capture`` subcommand
# -------------------------------------------------------------------


class TestCaptureParser:
    """Argument parsing for the ``capture`` subcommand."""

    def test_capture_defaults(self) -> None:
        args = parse_args(["capture"])
        assert args.command == "capture"
        assert args.method == "wgc"
        assert args.count == 5
        assert args.output == "output/captures"

    def test_capture_custom(self) -> None:
        args = parse_args(
            [
                "capture",
                "--method",
                "mss",
                "--count",
                "10",
                "--output",
                "my_dir",
            ]
        )
        assert args.method == "mss"
        assert args.count == 10
        assert args.output == "my_dir"


# -------------------------------------------------------------------
# ``detect`` subcommand
# -------------------------------------------------------------------


class TestDetectParser:
    """Argument parsing for the ``detect`` subcommand."""

    def test_detect_requires_image(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["detect"])

    def test_detect_positional(self) -> None:
        args = parse_args(["detect", "frame.png"])
        assert args.command == "detect"
        assert args.image_path == "frame.png"

    def test_detect_defaults(self) -> None:
        args = parse_args(["detect", "frame.png"])
        assert args.config == DEFAULT_CONFIG_PATH
        assert args.output is None
        assert args.threshold is None

    def test_detect_custom(self) -> None:
        args = parse_args(
            [
                "detect",
                "frame.png",
                "--config",
                "c.toml",
                "--output",
                "out.png",
                "--threshold",
                "0.5",
            ]
        )
        assert args.config == "c.toml"
        assert args.output == "out.png"
        assert args.threshold == 0.5


# -------------------------------------------------------------------
# ``export-model`` subcommand
# -------------------------------------------------------------------


class TestExportModelParser:
    """Argument parsing for the ``export-model`` subcommand."""

    def test_export_model_defaults(self) -> None:
        args = parse_args(["export-model"])
        assert args.command == "export-model"
        assert args.output == "models/yolo11n.onnx"
        assert args.input_size == 640

    def test_export_model_custom(self) -> None:
        args = parse_args(
            [
                "export-model",
                "--output",
                "custom.onnx",
                "--input-size",
                "320",
            ]
        )
        assert args.output == "custom.onnx"
        assert args.input_size == 320


# -------------------------------------------------------------------
# No subcommand
# -------------------------------------------------------------------


class TestNoCommand:
    """When no subcommand is given."""

    def test_no_command_returns_none(self) -> None:
        args = parse_args([])
        assert args.command is None


# -------------------------------------------------------------------
# Handler integration tests (mocked dependencies)
# -------------------------------------------------------------------


class TestHandleRun:
    """Test the ``run`` handler dispatches correctly."""

    @patch("semabot.app.cli._handle_run")
    def test_main_dispatches_run(self, mock_handler: MagicMock) -> None:
        main(["run", "--game", "test_game", "--dry-run"])
        mock_handler.assert_called_once()

    @patch("semabot.app.factory.create_bot")
    def test_handle_run_calls_factory(self, mock_create: MagicMock) -> None:
        mock_orch = MagicMock()
        mock_create.return_value = mock_orch
        main(["run", "--game", "test_game", "--dry-run", "--duration", "1"])
        mock_create.assert_called_once()
        mock_orch.run.assert_called_once()


class TestHandleCapture:
    """Test the ``capture`` handler."""

    @patch("semabot.app.cli._make_capture_source")
    def test_handle_capture_creates_source(self, mock_make: MagicMock, tmp_path: object) -> None:
        mock_source = MagicMock()
        mock_source.get_latest_frame.return_value = None
        mock_make.return_value = mock_source
        main(["capture", "--count", "1", "--output", str(tmp_path)])
        mock_make.assert_called_once_with("wgc")
        mock_source.start.assert_called_once()
        mock_source.stop.assert_called_once()

    def test_make_capture_source_unknown_raises(self) -> None:
        from semabot.app.cli import _make_capture_source

        with pytest.raises(ValueError, match="Unknown capture method"):
            _make_capture_source("unknown")


class TestHandleDetect:
    """Test the ``detect`` handler."""

    @patch("semabot.app.cli._handle_detect")
    def test_main_dispatches_detect(self, mock_handler: MagicMock) -> None:
        main(["detect", "test.png"])
        mock_handler.assert_called_once()


class TestHandleExportModel:
    """Test the ``export-model`` handler."""

    def test_export_model_prints_instructions(self, capsys: object) -> None:
        main(["export-model"])
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "export" in captured.out.lower() or "Export" in captured.out


class TestMainNoCommand:
    """Test main with no subcommand."""

    def test_no_command_exits(self) -> None:
        with pytest.raises(SystemExit):
            main([])
