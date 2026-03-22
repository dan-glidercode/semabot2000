"""Tests for GameplayRecorder."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from semabot.training.recorder import GameplayRecorder

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_frame_source(
    frame: np.ndarray | None = None,
) -> MagicMock:
    """Return a mock frame source that yields *frame*."""
    source = MagicMock()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    source.get_latest_frame.return_value = frame
    return source


def _fake_time_factory(start: float, step: float) -> tuple[list[int], object]:
    """Return a (counter, callable) pair for fake time."""
    state = [0]

    def fake_time() -> float:
        state[0] += 1
        return start + (state[0] - 1) * step

    return state, fake_time


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestRecordSavesFiles:
    """Verify frames are saved to disk."""

    def test_correct_number_of_files(self, tmp_path: Path) -> None:
        mock_cv2 = MagicMock()
        source = _make_frame_source()
        recorder = GameplayRecorder(
            frame_source=source,
            interval_ms=100,
            output_dir=str(tmp_path),
        )
        _, fake_time = _fake_time_factory(1000.0, 0.1)

        with patch.dict(sys.modules, {"cv2": mock_cv2}), patch(
            "semabot.training.recorder.time.time",
            side_effect=fake_time,
        ), patch("semabot.training.recorder.time.sleep"):
            count = recorder.record(duration_s=0.5)

        # time calls: 1->end_time(1000.5), 2->1000.1 (loop),
        # 3->1000.2, 4->1000.3, 5->1000.4, 6->1000.5 exits
        assert count == 4
        assert mock_cv2.imwrite.call_count == 4


class TestMetadataContent:
    """Verify metadata.json is written correctly."""

    def test_metadata_has_all_fields(self, tmp_path: Path) -> None:
        mock_cv2 = MagicMock()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        source = _make_frame_source(frame)
        recorder = GameplayRecorder(
            frame_source=source,
            interval_ms=200,
            output_dir=str(tmp_path),
        )
        _, fake_time = _fake_time_factory(1000.0, 0.2)

        with patch.dict(sys.modules, {"cv2": mock_cv2}), patch(
            "semabot.training.recorder.time.time",
            side_effect=fake_time,
        ), patch("semabot.training.recorder.time.sleep"):
            recorder.record(duration_s=0.5)

        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists()
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        assert "timestamp" in meta
        assert meta["interval_ms"] == 200
        assert meta["frame_count"] == 2
        assert meta["frame_width"] == 1280
        assert meta["frame_height"] == 720
        assert "duration" in meta


class TestFrameNumbering:
    """Verify frame files are numbered sequentially."""

    def test_sequential_numbering(self, tmp_path: Path) -> None:
        mock_cv2 = MagicMock()
        source = _make_frame_source()
        recorder = GameplayRecorder(
            frame_source=source,
            interval_ms=100,
            output_dir=str(tmp_path),
        )
        _, fake_time = _fake_time_factory(1000.0, 0.1)

        with patch.dict(sys.modules, {"cv2": mock_cv2}), patch(
            "semabot.training.recorder.time.time",
            side_effect=fake_time,
        ), patch("semabot.training.recorder.time.sleep"):
            recorder.record(duration_s=0.35)

        calls = mock_cv2.imwrite.call_args_list
        filenames = [Path(c[0][0]).name for c in calls]
        assert filenames == [
            "frame_000001.png",
            "frame_000002.png",
            "frame_000003.png",
        ]


class TestHandlesNoneFrames:
    """Verify None frames are skipped."""

    def test_none_frames_skipped(self, tmp_path: Path) -> None:
        mock_cv2 = MagicMock()
        source = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        source.get_latest_frame.side_effect = [
            None,
            frame,
            None,
            frame,
        ]
        recorder = GameplayRecorder(
            frame_source=source,
            interval_ms=100,
            output_dir=str(tmp_path),
        )
        _, fake_time = _fake_time_factory(1000.0, 0.1)

        with patch.dict(sys.modules, {"cv2": mock_cv2}), patch(
            "semabot.training.recorder.time.time",
            side_effect=fake_time,
        ), patch("semabot.training.recorder.time.sleep"):
            count = recorder.record(duration_s=0.5)

        assert count == 2
        assert mock_cv2.imwrite.call_count == 2


class TestOutputDirectoryCreated:
    """Verify the output directory is created if missing."""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        mock_cv2 = MagicMock()
        out = tmp_path / "nested" / "deep" / "output"
        source = _make_frame_source()
        recorder = GameplayRecorder(
            frame_source=source,
            interval_ms=100,
            output_dir=str(out),
        )
        _, fake_time = _fake_time_factory(1000.0, 0.1)

        with patch.dict(sys.modules, {"cv2": mock_cv2}), patch(
            "semabot.training.recorder.time.time",
            side_effect=fake_time,
        ), patch("semabot.training.recorder.time.sleep"):
            recorder.record(duration_s=0.15)

        assert (out / "images").is_dir()
