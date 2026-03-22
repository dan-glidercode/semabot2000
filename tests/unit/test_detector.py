"""Tests for YoloDetector — all ONNX interaction is mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from semabot.core.models import BoundingBox
from semabot.intelligence.detector import YoloDetector

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_raw_output(
    entries: list[tuple[float, float, float, float, int, float]],
    num_classes: int = 80,
    num_anchors: int = 8400,
) -> np.ndarray:
    """Build a fake YOLO11n output tensor.

    Each *entry* is ``(cx, cy, w, h, class_id, score)``.
    Returns shape ``(1, 4 + num_classes, num_anchors)``.
    """
    data = np.zeros((num_anchors, 4 + num_classes), dtype=np.float32)
    for i, (cx, cy, w, h, cid, score) in enumerate(entries):
        data[i, 0] = cx
        data[i, 1] = cy
        data[i, 2] = w
        data[i, 3] = h
        data[i, 4 + cid] = score
    # Transpose to (84, 8400) then add batch dim -> (1, 84, 8400)
    return data.T[np.newaxis].astype(np.float32)


def _make_detector(
    raw_output: np.ndarray,
    confidence_threshold: float = 0.35,
    nms_iou_threshold: float = 0.5,
) -> YoloDetector:
    """Create a YoloDetector with a mocked ONNX session."""
    detector = YoloDetector(
        model_path="fake.onnx",
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [raw_output]
    detector._session = mock_session  # type: ignore[assignment]
    return detector


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestYoloDetector:
    """YoloDetector decode, NMS, and filtering tests."""

    def test_single_detection(self) -> None:
        """One high-confidence detection is returned correctly."""
        raw = _make_raw_output(
            [
                (320.0, 240.0, 100.0, 80.0, 0, 0.9),
            ]
        )
        detector = _make_detector(raw)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        assert len(dets) == 1
        d = dets[0]
        assert d.class_name == "person"
        assert pytest.approx(d.confidence, abs=1e-5) == 0.9
        assert pytest.approx(d.bbox.x1, abs=1e-3) == 270.0
        assert pytest.approx(d.bbox.y1, abs=1e-3) == 200.0
        assert pytest.approx(d.bbox.x2, abs=1e-3) == 370.0
        assert pytest.approx(d.bbox.y2, abs=1e-3) == 280.0

    def test_multiple_non_overlapping(self) -> None:
        """Two well-separated detections both survive NMS."""
        raw = _make_raw_output(
            [
                (100.0, 100.0, 50.0, 50.0, 0, 0.8),
                (500.0, 500.0, 60.0, 60.0, 2, 0.7),
            ]
        )
        detector = _make_detector(raw)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        assert len(dets) == 2
        names = {d.class_name for d in dets}
        assert "person" in names
        assert "car" in names

    def test_confidence_threshold_filters(self) -> None:
        """Detections below confidence_threshold are discarded."""
        raw = _make_raw_output(
            [
                (100.0, 100.0, 50.0, 50.0, 0, 0.9),
                (300.0, 300.0, 50.0, 50.0, 1, 0.2),
            ]
        )
        detector = _make_detector(raw, confidence_threshold=0.35)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        assert len(dets) == 1
        assert dets[0].class_name == "person"

    def test_all_below_threshold(self) -> None:
        """If every detection is below the threshold, return []."""
        raw = _make_raw_output(
            [
                (100.0, 100.0, 50.0, 50.0, 0, 0.1),
                (300.0, 300.0, 50.0, 50.0, 1, 0.05),
            ]
        )
        detector = _make_detector(raw, confidence_threshold=0.35)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        assert dets == []

    def test_nms_suppresses_overlapping(self) -> None:
        """Highly overlapping boxes: only the best-scoring survives."""
        raw = _make_raw_output(
            [
                (200.0, 200.0, 100.0, 100.0, 0, 0.9),
                (205.0, 205.0, 100.0, 100.0, 0, 0.7),
            ]
        )
        detector = _make_detector(
            raw,
            confidence_threshold=0.35,
            nms_iou_threshold=0.5,
        )
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        assert len(dets) == 1
        assert pytest.approx(dets[0].confidence, abs=1e-5) == 0.9

    def test_nms_keeps_low_overlap(self) -> None:
        """Boxes that barely overlap both survive NMS."""
        raw = _make_raw_output(
            [
                (100.0, 100.0, 50.0, 50.0, 0, 0.9),
                (200.0, 200.0, 50.0, 50.0, 0, 0.85),
            ]
        )
        detector = _make_detector(
            raw,
            nms_iou_threshold=0.5,
        )
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        assert len(dets) == 2

    def test_class_id_mapping(self) -> None:
        """Various class IDs map to correct COCO names."""
        entries = [
            (100.0, 100.0, 50.0, 50.0, 0, 0.9),  # person
            (300.0, 100.0, 50.0, 50.0, 15, 0.8),  # cat
            (500.0, 100.0, 50.0, 50.0, 16, 0.7),  # dog
        ]
        raw = _make_raw_output(entries)
        detector = _make_detector(raw)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        name_map = {d.class_name: d for d in dets}
        assert "person" in name_map
        assert "cat" in name_map
        assert "dog" in name_map

    def test_detection_has_bounding_box(self) -> None:
        """Each Detection wraps a BoundingBox with correct coords."""
        raw = _make_raw_output(
            [
                (50.0, 60.0, 20.0, 30.0, 5, 0.95),
            ]
        )
        detector = _make_detector(raw)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)

        assert len(dets) == 1
        bbox = dets[0].bbox
        assert isinstance(bbox, BoundingBox)
        assert pytest.approx(bbox.x1, abs=1e-3) == 40.0
        assert pytest.approx(bbox.y1, abs=1e-3) == 45.0
        assert pytest.approx(bbox.x2, abs=1e-3) == 60.0
        assert pytest.approx(bbox.y2, abs=1e-3) == 75.0

    def test_empty_raw_output(self) -> None:
        """All-zero raw output produces no detections."""
        raw = np.zeros((1, 84, 8400), dtype=np.float32)
        detector = _make_detector(raw)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        dets = detector.detect(blob)
        assert dets == []

    def test_lazy_session_not_created_in_init(self) -> None:
        """ONNX session is None right after construction."""
        detector = YoloDetector(model_path="fake.onnx")
        assert detector._session is None

    def test_lazy_session_created_on_detect(self) -> None:
        """_ensure_session is called when detect() runs."""
        raw = _make_raw_output([])
        detector = _make_detector(raw)
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        detector.detect(blob)
        assert detector._session is not None


class TestYoloDetectorIoU:
    """Unit tests for the static _iou helper."""

    def test_identical_boxes(self) -> None:
        a = np.array([0.0, 0.0, 10.0, 10.0])
        assert pytest.approx(YoloDetector._iou(a, a)) == 1.0

    def test_no_overlap(self) -> None:
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([20.0, 20.0, 30.0, 30.0])
        assert YoloDetector._iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([5.0, 5.0, 15.0, 15.0])
        # Intersection: 5x5 = 25, union: 100+100-25 = 175
        assert pytest.approx(YoloDetector._iou(a, b), abs=1e-5) == 25.0 / 175.0
