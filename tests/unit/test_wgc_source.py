"""Tests for WGCFrameSource with mocked dependencies."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np

from semabot.capture.wgc_source import WGCFrameSource, _bgra_to_bgr

# -------------------------------------------------------------------
# BGRA -> BGR conversion
# -------------------------------------------------------------------


class TestBgraToBgr:
    """Unit tests for the _bgra_to_bgr helper."""

    def test_drops_alpha_channel(self) -> None:
        bgra = np.zeros((100, 200, 4), dtype=np.uint8)
        bgra[:, :, 0] = 10  # B
        bgra[:, :, 1] = 20  # G
        bgra[:, :, 2] = 30  # R
        bgra[:, :, 3] = 255  # A

        bgr = _bgra_to_bgr(bgra)

        assert bgr.shape == (100, 200, 3)
        assert np.all(bgr[:, :, 0] == 10)
        assert np.all(bgr[:, :, 1] == 20)
        assert np.all(bgr[:, :, 2] == 30)

    def test_preserves_pixel_values(self) -> None:
        bgra = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        bgr = _bgra_to_bgr(bgra)
        np.testing.assert_array_equal(bgr, bgra[:, :, :3])

    def test_single_pixel(self) -> None:
        bgra = np.array([[[100, 150, 200, 255]]], dtype=np.uint8)
        bgr = _bgra_to_bgr(bgra)
        assert bgr.shape == (1, 1, 3)
        np.testing.assert_array_equal(bgr[0, 0], [100, 150, 200])


# -------------------------------------------------------------------
# WGCFrameSource construction & lifecycle
# -------------------------------------------------------------------


class TestWGCFrameSourceConstruction:
    """Tests for WGCFrameSource init and basic state."""

    def test_initial_state(self) -> None:
        src = WGCFrameSource(window_title="TestWindow")
        assert src._window_title == "TestWindow"
        assert src._latest_frame is None
        assert src._thread is None

    def test_get_latest_frame_returns_none_before_start(
        self,
    ) -> None:
        src = WGCFrameSource(window_title="TestWindow")
        assert src.get_latest_frame() is None


class TestWGCFrameSourceLifecycle:
    """Tests for start / stop with mocked WGC and win32gui."""

    @patch("semabot.capture.wgc_source.WGCFrameSource._capture_loop")
    def test_start_creates_thread(self, mock_loop: MagicMock) -> None:
        src = WGCFrameSource(window_title="TestWindow")
        src.start()

        assert src._thread is not None
        assert src._thread.is_alive() or mock_loop.called
        # Clean up
        src._stop_event.set()
        src._thread.join(timeout=1.0)

    @patch("semabot.capture.wgc_source.WGCFrameSource._capture_loop")
    def test_stop_joins_thread(self, mock_loop: MagicMock) -> None:
        src = WGCFrameSource(window_title="TestWindow")
        src.start()
        src.stop()

        assert src._thread is None
        assert src._stop_event.is_set()

    @patch("semabot.capture.wgc_source.WGCFrameSource._capture_loop")
    def test_stop_without_start_is_safe(self, mock_loop: MagicMock) -> None:
        src = WGCFrameSource(window_title="TestWindow")
        src.stop()  # Should not raise


# -------------------------------------------------------------------
# Frame buffer (thread-safe storage)
# -------------------------------------------------------------------


def _simulate_frame_arrival(src: WGCFrameSource, bgra: np.ndarray) -> None:
    """Simulate what the WGC callback does: convert BGRA->BGR and store."""
    bgr = _bgra_to_bgr(bgra)
    with src._lock:
        src._latest_frame = bgr


class TestWGCFrameBuffer:
    """Tests for the frame buffer and thread-safe storage."""

    def test_callback_stores_bgr_frame(self) -> None:
        src = WGCFrameSource(window_title="TestWindow")
        bgra = np.zeros((480, 640, 4), dtype=np.uint8)
        bgra[:, :, 2] = 128  # R channel

        _simulate_frame_arrival(src, bgra)

        frame = src.get_latest_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert np.all(frame[:, :, 2] == 128)

    def test_latest_frame_is_most_recent(self) -> None:
        src = WGCFrameSource(window_title="TestWindow")

        frame1 = np.zeros((100, 100, 4), dtype=np.uint8)
        frame1[:, :, 0] = 10
        frame2 = np.zeros((100, 100, 4), dtype=np.uint8)
        frame2[:, :, 0] = 20

        _simulate_frame_arrival(src, frame1)
        _simulate_frame_arrival(src, frame2)

        latest = src.get_latest_frame()
        assert latest is not None
        assert np.all(latest[:, :, 0] == 20)

    def test_thread_safety_of_buffer(self) -> None:
        """Multiple threads writing/reading should not crash."""
        src = WGCFrameSource(window_title="TestWindow")
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    bgra = np.full((10, 10, 4), i % 256, dtype=np.uint8)
                    _simulate_frame_arrival(src, bgra)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(50):
                    src.get_latest_frame()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == []
