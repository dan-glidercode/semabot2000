"""Integration test: YoloPreprocessor -> YoloDetector -> GameStateBuilder."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from semabot.core.models import BoundingBox, GameState
from semabot.intelligence.detector import YoloDetector
from semabot.intelligence.preprocessor import YoloPreprocessor
from semabot.intelligence.state_builder import GameStateBuilder

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _make_raw_output(
    entries: list[tuple[float, float, float, float, int, float]],
    num_classes: int = 80,
    num_anchors: int = 8400,
) -> np.ndarray:
    """Build a fake YOLO11n output tensor.

    Each *entry* is ``(cx, cy, w, h, class_id, score)``.
    Returns shape ``(1, 4 + num_classes, num_anchors)``.
    """
    data = np.zeros(
        (num_anchors, 4 + num_classes),
        dtype=np.float32,
    )
    for i, (cx, cy, w, h, cid, score) in enumerate(entries):
        data[i, 0] = cx
        data[i, 1] = cy
        data[i, 2] = w
        data[i, 3] = h
        data[i, 4 + cid] = score
    return data.T[np.newaxis].astype(np.float32)


def _make_mock_session(
    raw_output: np.ndarray,
) -> MagicMock:
    """Return a mocked ONNX InferenceSession."""
    session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    session.get_inputs.return_value = [mock_input]
    session.run.return_value = [raw_output]
    return session


def _build_synthetic_frame() -> np.ndarray:
    """Create a synthetic 800x600 BGR image with structure."""
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    # Add a bright rectangle to give it some structure
    frame[100:200, 150:300] = [0, 255, 0]
    frame[300:500, 400:600] = [0, 0, 255]
    return frame


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------


class TestDetectionPipeline:
    """Full detection flow: preprocess -> detect -> build state."""

    def test_single_detection_pipeline(self) -> None:
        """One known detection flows through the full pipeline."""
        frame = _build_synthetic_frame()

        # Step 1: preprocess
        preprocessor = YoloPreprocessor(input_size=640)
        blob = preprocessor.process(frame)

        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32

        # Step 2: detect with mocked session
        raw = _make_raw_output(
            [
                (320.0, 240.0, 100.0, 80.0, 0, 0.9),
            ]
        )
        detector = YoloDetector(
            model_path="fake.onnx",
            confidence_threshold=0.35,
        )
        detector._session = _make_mock_session(raw)

        detections = detector.detect(blob)

        assert len(detections) == 1
        assert detections[0].class_name == "person"
        assert detections[0].confidence == pytest.approx(0.9)

        # Step 3: build game state
        builder = GameStateBuilder(confidence_threshold=0.35)
        state = builder.build(detections, frame)

        assert isinstance(state, GameState)
        assert len(state.detections) == 1
        assert state.detections[0].class_name == "person"
        assert state.frame_width == 800
        assert state.frame_height == 600

    def test_two_detections_pipeline(self) -> None:
        """Two known detections pass through the full pipeline."""
        frame = _build_synthetic_frame()

        preprocessor = YoloPreprocessor(input_size=640)
        blob = preprocessor.process(frame)

        raw = _make_raw_output(
            [
                (100.0, 100.0, 50.0, 50.0, 0, 0.8),
                (500.0, 400.0, 60.0, 60.0, 15, 0.7),
            ]
        )
        detector = YoloDetector(
            model_path="fake.onnx",
            confidence_threshold=0.35,
        )
        detector._session = _make_mock_session(raw)

        detections = detector.detect(blob)

        assert len(detections) == 2
        names = {d.class_name for d in detections}
        assert "person" in names
        assert "cat" in names

        builder = GameStateBuilder(confidence_threshold=0.35)
        state = builder.build(detections, frame)

        assert len(state.detections) == 2
        state_names = {d.class_name for d in state.detections}
        assert "person" in state_names
        assert "cat" in state_names

    def test_bbox_coords_survive_pipeline(self) -> None:
        """Bounding box coordinates are preserved through the flow."""
        frame = _build_synthetic_frame()

        preprocessor = YoloPreprocessor(input_size=640)
        blob = preprocessor.process(frame)

        raw = _make_raw_output(
            [
                (200.0, 150.0, 80.0, 60.0, 2, 0.85),
            ]
        )
        detector = YoloDetector(
            model_path="fake.onnx",
            confidence_threshold=0.3,
        )
        detector._session = _make_mock_session(raw)

        detections = detector.detect(blob)
        builder = GameStateBuilder(confidence_threshold=0.3)
        state = builder.build(detections, frame)

        det = state.detections[0]
        assert det.class_name == "car"
        assert isinstance(det.bbox, BoundingBox)
        assert det.bbox.x1 == pytest.approx(160.0)
        assert det.bbox.y1 == pytest.approx(120.0)
        assert det.bbox.x2 == pytest.approx(240.0)
        assert det.bbox.y2 == pytest.approx(180.0)

    def test_no_detections_pipeline(self) -> None:
        """Zero detections produce an empty GameState."""
        frame = _build_synthetic_frame()

        preprocessor = YoloPreprocessor(input_size=640)
        blob = preprocessor.process(frame)

        raw = np.zeros((1, 84, 8400), dtype=np.float32)
        detector = YoloDetector(
            model_path="fake.onnx",
            confidence_threshold=0.35,
        )
        detector._session = _make_mock_session(raw)

        detections = detector.detect(blob)
        builder = GameStateBuilder(confidence_threshold=0.35)
        state = builder.build(detections, frame)

        assert len(state.detections) == 0
        assert state.frame_width == 800
        assert state.frame_height == 600
