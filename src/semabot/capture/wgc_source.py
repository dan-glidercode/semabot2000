"""WGC-based frame source using windows-capture library.

Runs Windows Graphics Capture in a background thread and exposes
the latest frame via a thread-safe buffer.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def _bgra_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert a BGRA frame to BGR by dropping the alpha channel."""
    return frame[:, :, :3]


class WGCFrameSource:
    """FrameSource implementation backed by Windows Graphics Capture.

    Implements the ``FrameSource`` protocol defined in
    ``semabot.core.protocols``.
    """

    def __init__(self, window_title: str) -> None:
        self._window_title = window_title
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- FrameSource protocol ----------------------------------------

    def start(self) -> None:
        """Launch background capture thread targeting *window_title*."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="wgc-capture",
        )
        self._thread.start()
        logger.info(
            "WGC capture started for window '%s'",
            self._window_title,
        )

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recent BGR frame, or *None*."""
        with self._lock:
            return self._latest_frame

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("WGC capture stopped.")

    # -- internals ---------------------------------------------------

    def _on_frame_arrived(self, frame: np.ndarray) -> None:
        """Callback invoked by WindowsCapture on each new frame."""
        bgr = _bgra_to_bgr(frame)
        with self._lock:
            self._latest_frame = bgr

    def _capture_loop(self) -> None:
        """Background thread: find the window and run WGC capture."""
        import win32gui  # noqa: WPS433 — lazy import
        from windows_capture import WindowsCapture  # noqa: WPS433

        hwnd = win32gui.FindWindow(None, self._window_title)
        if not hwnd:
            logger.error("Window '%s' not found.", self._window_title)
            return

        capture = WindowsCapture(
            hwnd=hwnd,
            on_frame_arrived=self._on_frame_arrived,
        )
        logger.debug("WindowsCapture object created for hwnd=%s", hwnd)

        # Block until stop is requested.
        capture.start()
        self._stop_event.wait()
        capture.stop()
