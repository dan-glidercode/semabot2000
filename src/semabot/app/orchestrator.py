"""BotOrchestrator — main loop that wires all pipeline components."""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING

from semabot.app.metrics import PipelineMetrics

if TYPE_CHECKING:
    import numpy as np

    from semabot.core.config import BotConfig
    from semabot.core.models import Detection
    from semabot.core.protocols import (
        DecisionEngine,
        Detector,
        FrameSource,
        InputController,
        Preprocessor,
        StateBuilder,
    )

logger = logging.getLogger(__name__)


class BotOrchestrator:
    """Coordinates the capture-detect-decide-act pipeline.

    The orchestrator owns no business logic.  It captures a frame,
    preprocesses it, detects objects, builds game state, decides
    on an action, and executes it — delegating each step to a
    pluggable component.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        preprocessor: Preprocessor,
        detector: Detector,
        state_builder: StateBuilder,
        decision_engine: DecisionEngine,
        input_controller: InputController,
        config: BotConfig,
        save_detections: bool = False,
        save_dir: str = "output/detections",
    ) -> None:
        self._frame_source = frame_source
        self._preprocessor = preprocessor
        self._detector = detector
        self._state_builder = state_builder
        self._decision_engine = decision_engine
        self._input_controller = input_controller
        self._config = config
        self._running = False
        self._metrics = PipelineMetrics()
        self._save_detections = save_detections
        self._save_dir = save_dir

    # -- public API ---------------------------------------------------

    def run(self, duration: float | None = None) -> None:
        """Run the main pipeline loop.

        Parameters
        ----------
        duration:
            If set, stop after this many seconds.  If *None*,
            run until :meth:`stop` is called.
        """
        self._install_signal_handler()
        self._log_startup()
        self._running = True
        self._frame_source.start()
        try:
            self._loop(duration)
        finally:
            self._frame_source.stop()
            self._input_controller.release_all()

    def stop(self) -> None:
        """Signal the main loop to exit."""
        self._running = False

    # -- internals ----------------------------------------------------

    def _install_signal_handler(self) -> None:
        """Register SIGINT handler that calls :meth:`stop`."""
        try:
            signal.signal(signal.SIGINT, self._on_sigint)
        except (OSError, ValueError):
            # Cannot set signal handler from a non-main thread
            pass

    def _on_sigint(
        self,
        signum: int,
        frame: object,
    ) -> None:
        """Handle SIGINT by requesting a clean shutdown."""
        logger.info("SIGINT received — stopping.")
        self.stop()

    def _log_startup(self) -> None:
        """Log startup configuration at INFO level."""
        logger.info(
            "Starting orchestrator — " "model=%s window=%s provider=%s",
            self._config.detection.model_path,
            self._config.capture.window_title,
            self._config.detection.provider,
        )

    def _loop(self, duration: float | None) -> None:
        """Core loop: capture, process, decide, act."""
        frame_count = 0
        start_time = time.monotonic()
        last_fps_log = start_time
        last_metrics_log = start_time

        while self._running:
            if self._should_stop(start_time, duration):
                break

            if not self._tick_safe(frame_count):
                time.sleep(0.001)
                continue

            frame_count += 1
            now = time.monotonic()
            last_fps_log = self._maybe_log_fps(
                frame_count,
                start_time,
                last_fps_log,
            )
            last_metrics_log = self._maybe_log_metrics(
                now,
                last_metrics_log,
            )

    def _tick_safe(self, frame_count: int) -> bool:
        """Run one pipeline tick, catching non-fatal errors."""
        try:
            return self._tick(frame_count)
        except Exception:
            logger.exception(
                "Non-fatal error in pipeline tick",
            )
            return False

    def _tick(self, frame_count: int) -> bool:
        """Run one pipeline iteration.

        Return True if a frame was processed.
        """
        self._metrics.begin_frame()

        frame = self._frame_source.get_latest_frame()
        self._metrics.record("capture")

        if frame is None:
            return False

        blob = self._preprocessor.process(frame)
        self._metrics.record("preprocess")

        detections = self._detector.detect(blob)
        self._metrics.record("detect")

        state = self._state_builder.build(detections, frame)
        self._metrics.record("postprocess")

        action = self._decision_engine.decide(state)
        self._metrics.record("decide")

        self._input_controller.execute(action)
        self._metrics.record("act")

        self._metrics.end_frame()

        logger.debug(
            "Frame: %d detections — %s",
            len(state.detections),
            action.description or "(idle)",
        )

        if self._save_detections:
            self._maybe_save_frame(
                frame,
                detections,
                frame_count,
            )

        return True

    def _maybe_save_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_count: int,
    ) -> None:
        """Save annotated frame every 10th frame."""
        if frame_count % 10 != 0:
            return
        self._save_annotated(frame, detections, frame_count)

    def _save_annotated(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_count: int,
    ) -> None:
        """Draw bboxes on *frame* and write to *save_dir*."""
        import cv2

        out_dir = Path(self._save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()
        for det in detections:
            x1 = int(det.bbox.x1)
            y1 = int(det.bbox.y1)
            x2 = int(det.bbox.x2)
            y2 = int(det.bbox.y2)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        path = out_dir / f"frame_{frame_count:06d}.png"
        cv2.imwrite(str(path), annotated)
        logger.debug("Saved detection frame: %s", path)

    @staticmethod
    def _should_stop(
        start_time: float,
        duration: float | None,
    ) -> bool:
        """Return True when the duration budget is exhausted."""
        if duration is None:
            return False
        return (time.monotonic() - start_time) >= duration

    @staticmethod
    def _maybe_log_fps(
        frame_count: int,
        start_time: float,
        last_fps_log: float,
    ) -> float:
        """Log FPS every 5 seconds and return updated timestamp."""
        now = time.monotonic()
        if now - last_fps_log < 5.0:
            return last_fps_log

        elapsed = now - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            logger.info(
                "FPS: %.1f (%d frames / %.1fs)",
                fps,
                frame_count,
                elapsed,
            )
        return now

    def _maybe_log_metrics(
        self,
        now: float,
        last_log: float,
    ) -> float:
        """Log pipeline metrics every 5 seconds."""
        if now - last_log < 5.0:
            return last_log
        self._metrics.log_summary()
        return now
