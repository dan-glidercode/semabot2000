"""BotOrchestrator — main loop that wires all pipeline components."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semabot.core.config import BotConfig
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
    ) -> None:
        self._frame_source = frame_source
        self._preprocessor = preprocessor
        self._detector = detector
        self._state_builder = state_builder
        self._decision_engine = decision_engine
        self._input_controller = input_controller
        self._config = config
        self._running = False

    # -- public API ---------------------------------------------------

    def run(self, duration: float | None = None) -> None:
        """Run the main pipeline loop.

        Parameters
        ----------
        duration:
            If set, stop after this many seconds.  If *None*,
            run until :meth:`stop` is called.
        """
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

    def _loop(self, duration: float | None) -> None:
        """Core loop: capture, process, decide, act."""
        frame_count = 0
        start_time = time.monotonic()
        last_fps_log = start_time

        while self._running:
            if self._should_stop(start_time, duration):
                break

            if not self._tick():
                continue

            frame_count += 1
            last_fps_log = self._maybe_log_fps(
                frame_count,
                start_time,
                last_fps_log,
            )

    def _tick(self) -> bool:
        """Run one pipeline iteration.  Return True if a frame was processed."""
        frame = self._frame_source.get_latest_frame()
        if frame is None:
            return False

        blob = self._preprocessor.process(frame)
        detections = self._detector.detect(blob)
        state = self._state_builder.build(detections, frame)
        action = self._decision_engine.decide(state)
        self._input_controller.execute(action)
        return True

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
