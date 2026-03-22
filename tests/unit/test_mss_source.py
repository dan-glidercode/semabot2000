"""Tests for MSSFrameSource with mocked mss dependency."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from semabot.capture.mss_source import MSSFrameSource, _bgra_to_bgr

# -------------------------------------------------------------------
# BGRA -> BGR conversion (shared helper)
# -------------------------------------------------------------------


class TestBgraToBgr:
    """Unit tests for the _bgra_to_bgr helper."""

    def test_drops_alpha_channel(self) -> None:
        bgra = np.zeros((50, 80, 4), dtype=np.uint8)
        bgra[:, :, 3] = 255
        bgr = _bgra_to_bgr(bgra)
        assert bgr.shape == (50, 80, 3)

    def test_preserves_pixel_values(self) -> None:
        bgra = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
        bgr = _bgra_to_bgr(bgra)
        np.testing.assert_array_equal(bgr, bgra[:, :, :3])


# -------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------


class TestMSSFrameSourceConstruction:
    """Tests for MSSFrameSource init."""

    def test_stores_region(self) -> None:
        region = {
            "left": 0,
            "top": 0,
            "width": 1920,
            "height": 1080,
        }
        src = MSSFrameSource(window_region=region)
        assert src._region == region

    def test_region_copies_values(self) -> None:
        region = {
            "left": 10,
            "top": 20,
            "width": 800,
            "height": 600,
        }
        src = MSSFrameSource(window_region=region)
        region["left"] = 999
        assert src._region["left"] == 10


# -------------------------------------------------------------------
# start / stop are no-ops
# -------------------------------------------------------------------


class TestMSSFrameSourceLifecycle:
    """Verify start/stop are harmless no-ops."""

    def test_start_does_not_raise(self) -> None:
        src = MSSFrameSource(
            window_region={
                "left": 0,
                "top": 0,
                "width": 100,
                "height": 100,
            }
        )
        src.start()  # should not raise

    def test_stop_does_not_raise(self) -> None:
        src = MSSFrameSource(
            window_region={
                "left": 0,
                "top": 0,
                "width": 100,
                "height": 100,
            }
        )
        src.stop()  # should not raise


# -------------------------------------------------------------------
# get_latest_frame with mocked mss
# -------------------------------------------------------------------


def _make_mock_mss(fake_bgra: np.ndarray) -> MagicMock:
    """Build a mock ``mss`` module whose grab returns *fake_bgra*.

    The mock's ``mss()`` context manager yields an object whose
    ``grab()`` returns a screenshot-like object that, when passed
    to ``np.array()``, produces *fake_bgra*.
    """
    mock_sct = MagicMock()
    mock_sct.__enter__ = MagicMock(return_value=mock_sct)
    mock_sct.__exit__ = MagicMock(return_value=False)

    # np.array(shot) must give fake_bgra.  We make the shot
    # quack like something np.array can consume.
    mock_shot = np.asarray(fake_bgra)
    mock_sct.grab.return_value = mock_shot

    mock_mss_mod = MagicMock()
    mock_mss_mod.mss.return_value = mock_sct
    return mock_mss_mod


class TestMSSFrameSourceCapture:
    """Tests for get_latest_frame with mocked mss."""

    def test_returns_bgr_frame(self) -> None:
        region = {
            "left": 0,
            "top": 0,
            "width": 4,
            "height": 3,
        }
        src = MSSFrameSource(window_region=region)

        fake_bgra = np.zeros((3, 4, 4), dtype=np.uint8)
        fake_bgra[:, :, 0] = 50  # B
        fake_bgra[:, :, 1] = 100  # G
        fake_bgra[:, :, 2] = 150  # R
        fake_bgra[:, :, 3] = 255  # A

        mock_mss_mod = _make_mock_mss(fake_bgra)

        with patch.dict(sys.modules, {"mss": mock_mss_mod}):
            frame = src.get_latest_frame()

        assert frame is not None
        assert frame.shape == (3, 4, 3)
        assert np.all(frame[:, :, 0] == 50)
        assert np.all(frame[:, :, 1] == 100)
        assert np.all(frame[:, :, 2] == 150)

    def test_grab_called_with_region(self) -> None:
        region = {
            "left": 10,
            "top": 20,
            "width": 800,
            "height": 600,
        }
        src = MSSFrameSource(window_region=region)

        fake_bgra = np.zeros((600, 800, 4), dtype=np.uint8)
        mock_mss_mod = _make_mock_mss(fake_bgra)

        with patch.dict(sys.modules, {"mss": mock_mss_mod}):
            src.get_latest_frame()

        mock_sct = mock_mss_mod.mss.return_value
        mock_sct.grab.assert_called_once_with(region)

    def test_each_call_grabs_fresh_frame(self) -> None:
        region = {
            "left": 0,
            "top": 0,
            "width": 2,
            "height": 2,
        }
        src = MSSFrameSource(window_region=region)

        fake_bgra = np.zeros((2, 2, 4), dtype=np.uint8)
        mock_mss_mod = _make_mock_mss(fake_bgra)

        with patch.dict(sys.modules, {"mss": mock_mss_mod}):
            src.get_latest_frame()
            src.get_latest_frame()

        mock_sct = mock_mss_mod.mss.return_value
        assert mock_sct.grab.call_count == 2
