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
    if frame.ndim == 3 and frame.shape[2] == 4:
        return frame[:, :, :3].copy()
    return frame


class WGCFrameSource:
    """FrameSource implementation backed by Windows Graphics Capture.

    Uses the ``windows-capture`` library which provides an event-driven
    API via ``@capture.event`` decorators.  The capture runs in a
    background thread (``capture.start()`` blocks), and frames arrive
    via the ``on_frame_arrived`` callback.
    """

    def __init__(self, window_title: str) -> None:
        self._window_title = window_title
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_control = None

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
        # If we have a capture_control reference, tell it to stop
        ctrl = self._capture_control
        if ctrl is not None:
            try:
                ctrl.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("WGC capture stopped.")

    # -- internals ---------------------------------------------------

    _MAX_RETRIES: int = 3
    _RETRY_DELAY: float = 2.0

    def _create_capture_with_retry(
        self,
        capture_cls: type,
    ) -> object:
        """Try to create a WindowsCapture, retrying on failure.

        Retries up to ``_MAX_RETRIES`` times with a
        ``_RETRY_DELAY`` second sleep between attempts.
        Raises the last exception if all attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return capture_cls(
                    cursor_capture=None,
                    draw_border=None,
                    monitor_index=None,
                    window_name=self._window_title,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Window '%s' not found (attempt %d/%d)" " — retrying in %.0fs",
                    self._window_title,
                    attempt,
                    self._MAX_RETRIES,
                    self._RETRY_DELAY,
                )
                import time as _time

                _time.sleep(self._RETRY_DELAY)

        msg = f"Window '{self._window_title}' not found after " f"{self._MAX_RETRIES} retries"
        raise RuntimeError(msg) from last_exc

    def _capture_loop(self) -> None:  # pragma: no cover
        """Background thread: create WindowsCapture and run it."""
        from windows_capture import (
            Frame,
            InternalCaptureControl,
            WindowsCapture,
        )

        capture = self._create_capture_with_retry(
            WindowsCapture,
        )

        source = self  # reference for closures

        @capture.event
        def on_frame_arrived(
            frame: Frame,
            capture_control: InternalCaptureControl,
        ) -> None:
            source._capture_control = capture_control
            if source._stop_event.is_set():
                capture_control.stop()
                return
            bgr = _bgra_to_bgr(frame.frame_buffer)
            with source._lock:
                source._latest_frame = bgr

        @capture.event
        def on_closed() -> None:
            logger.debug("WGC capture session closed.")

        logger.debug(
            "WindowsCapture created for window '%s'",
            self._window_title,
        )
        # capture.start() blocks until on_closed or capture_control.stop()
        capture.start()
