"""MSS-based frame source using python-mss (GDI BitBlt fallback).

Synchronous capture — each call to ``get_latest_frame`` grabs a fresh
screenshot of the configured region.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def _bgra_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert a BGRA frame to BGR by dropping the alpha channel."""
    return frame[:, :, :3]


class MSSFrameSource:
    """FrameSource implementation backed by python-mss.

    Implements the ``FrameSource`` protocol defined in
    ``semabot.core.protocols``.

    Unlike :class:`WGCFrameSource`, capture is synchronous — there is
    no background thread.  ``start()`` and ``stop()`` are no-ops.
    """

    def __init__(self, window_region: dict) -> None:
        self._region = {
            "left": window_region["left"],
            "top": window_region["top"],
            "width": window_region["width"],
            "height": window_region["height"],
        }

    # -- FrameSource protocol ----------------------------------------

    def start(self) -> None:
        """No-op — MSS capture is synchronous."""
        logger.debug("MSSFrameSource.start() called (no-op).")

    def get_latest_frame(self) -> np.ndarray | None:
        """Grab a screenshot of the configured region and return BGR."""
        import mss  # noqa: WPS433 — lazy import
        import numpy as np  # noqa: WPS433

        with mss.mss() as sct:
            shot = sct.grab(self._region)
            frame_bgra = np.array(shot)
        return _bgra_to_bgr(frame_bgra)

    def stop(self) -> None:
        """No-op — nothing to tear down."""
        logger.debug("MSSFrameSource.stop() called (no-op).")
