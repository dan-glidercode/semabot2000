"""Gameplay recorder — captures frames at regular intervals."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class GameplayRecorder:
    """Records gameplay frames from a frame source at fixed intervals.

    Parameters
    ----------
    frame_source:
        Any object with a ``get_latest_frame()`` method that returns
        a numpy array or ``None``.
    interval_ms:
        Milliseconds between frame captures.
    output_dir:
        Base directory for recorded frames and metadata.
    """

    def __init__(
        self,
        frame_source: object,
        interval_ms: int = 500,
        output_dir: str = "datasets/recording",
    ) -> None:
        self._frame_source = frame_source
        self._interval_ms = interval_ms
        self._output_dir = output_dir

    def record(self, duration_s: float) -> int:
        """Capture frames for *duration_s* seconds.

        Returns the number of frames saved.
        """
        import cv2

        images_dir = Path(self._output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        interval_s = self._interval_ms / 1000.0
        end_time = time.time() + duration_s
        frame_count = 0
        frame_width = 0
        frame_height = 0

        while time.time() < end_time:
            frame = self._frame_source.get_latest_frame()  # type: ignore[union-attr]
            if frame is not None:
                frame_count += 1
                frame_height, frame_width = (
                    frame.shape[0],
                    frame.shape[1],
                )
                filename = f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(images_dir / filename), frame)
                if frame_count % 10 == 0:
                    logger.info("Recorded %d frames", frame_count)
            time.sleep(interval_s)

        self._write_metadata(frame_count, frame_width, frame_height)
        logger.info("Recording complete: %d frames saved", frame_count)
        return frame_count

    def _write_metadata(
        self,
        frame_count: int,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Write recording metadata to *output_dir/metadata.json*."""
        metadata = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "duration": self._interval_ms * frame_count / 1000.0,
            "interval_ms": self._interval_ms,
            "frame_count": frame_count,
            "frame_width": frame_width,
            "frame_height": frame_height,
        }
        meta_path = Path(self._output_dir) / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
