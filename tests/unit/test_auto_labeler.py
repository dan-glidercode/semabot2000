"""Tests for AutoLabeler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from semabot.core.models import BoundingBox, Detection
from semabot.training.auto_labeler import AutoLabeler

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_mock_preprocessor(
    input_size: int = 640,
) -> MagicMock:
    """Return a mock preprocessor that returns a blob."""
    pp = MagicMock()
    blob = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    pp.process.return_value = blob
    return pp


def _make_detections(
    class_name: str = "person",
    confidence: float = 0.8,
    bbox: BoundingBox | None = None,
) -> list[Detection]:
    """Build a single-element detection list."""
    if bbox is None:
        bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    return [
        Detection(
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
        )
    ]


def _create_test_images(images_dir: Path, count: int = 3) -> None:
    """Write tiny PNG-like files so glob('*.png') finds them."""
    import cv2

    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"frame_{i:06d}.png"), img)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestLabelOutputFormat:
    """Verify YOLO label format: class_id cx cy w h."""

    def test_normalized_values(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "images", count=1)
        det = Detection(
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(x1=64.0, y1=64.0, x2=192.0, y2=192.0),
        )
        pp = _make_mock_preprocessor(input_size=640)
        detector = MagicMock()
        detector.detect.return_value = [det]

        labeler = AutoLabeler(
            detector=detector,
            preprocessor=pp,
            class_map={"person": 0},
        )
        labeler.label_dataset(str(tmp_path / "images"))

        label_file = tmp_path / "labels" / "frame_000001.txt"
        assert label_file.exists()
        parts = label_file.read_text().strip().split()
        assert len(parts) == 5
        class_id = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
        assert class_id == 0
        # All values should be 0-1 (normalized by input_size)
        for val in (cx, cy, w, h):
            assert 0.0 <= val <= 1.0


class TestClassMapping:
    """Verify custom and auto-assigned class maps."""

    def test_custom_class_map(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "images", count=1)
        det = Detection(
            class_name="car",
            confidence=0.9,
            bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        )
        pp = _make_mock_preprocessor()
        detector = MagicMock()
        detector.detect.return_value = [det]

        labeler = AutoLabeler(
            detector=detector,
            preprocessor=pp,
            class_map={"car": 5},
        )
        labeler.label_dataset(str(tmp_path / "images"))

        label_file = tmp_path / "labels" / "frame_000001.txt"
        line = label_file.read_text().strip()
        assert line.startswith("5 ")

    def test_auto_assigned_ids(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "images", count=1)
        det_a = Detection(
            class_name="cat",
            confidence=0.9,
            bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        )
        det_b = Detection(
            class_name="dog",
            confidence=0.8,
            bbox=BoundingBox(x1=60.0, y1=60.0, x2=90.0, y2=90.0),
        )
        pp = _make_mock_preprocessor()
        detector = MagicMock()
        detector.detect.return_value = [det_a, det_b]

        labeler = AutoLabeler(detector=detector, preprocessor=pp)
        labeler.label_dataset(str(tmp_path / "images"))

        label_file = tmp_path / "labels" / "frame_000001.txt"
        lines = label_file.read_text().strip().split("\n")
        ids = [int(line.split()[0]) for line in lines]
        assert ids == [0, 1]


class TestThresholdFiltering:
    """Verify detections below threshold are excluded."""

    def test_below_threshold_excluded(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "images", count=1)
        high = Detection(
            class_name="person",
            confidence=0.8,
            bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        )
        low = Detection(
            class_name="car",
            confidence=0.1,
            bbox=BoundingBox(x1=60.0, y1=60.0, x2=90.0, y2=90.0),
        )
        pp = _make_mock_preprocessor()
        detector = MagicMock()
        detector.detect.return_value = [high, low]

        labeler = AutoLabeler(detector=detector, preprocessor=pp)
        labeler.label_dataset(str(tmp_path / "images"), threshold=0.5)

        label_file = tmp_path / "labels" / "frame_000001.txt"
        content = label_file.read_text().strip()
        lines = [line for line in content.split("\n") if line]
        assert len(lines) == 1
        assert lines[0].startswith("0 ")


class TestDataYaml:
    """Verify data.yaml is written correctly."""

    def test_data_yaml_content(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "images", count=1)
        det = Detection(
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        )
        pp = _make_mock_preprocessor()
        detector = MagicMock()
        detector.detect.return_value = [det]

        labeler = AutoLabeler(
            detector=detector,
            preprocessor=pp,
            class_map={"person": 0},
        )
        labeler.label_dataset(str(tmp_path / "images"))

        yaml_path = tmp_path / "data.yaml"
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "train:" in content
        assert "val:" in content
        assert "nc: 1" in content
        assert "person" in content


class TestNoDetections:
    """Verify images with no detections produce empty files."""

    def test_empty_label_file(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "images", count=1)
        pp = _make_mock_preprocessor()
        detector = MagicMock()
        detector.detect.return_value = []

        labeler = AutoLabeler(detector=detector, preprocessor=pp)
        labeler.label_dataset(str(tmp_path / "images"))

        label_file = tmp_path / "labels" / "frame_000001.txt"
        assert label_file.exists()
        assert label_file.read_text().strip() == ""
