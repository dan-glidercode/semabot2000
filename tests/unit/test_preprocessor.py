"""Tests for YoloPreprocessor."""

from __future__ import annotations

import numpy as np
import pytest

from semabot.intelligence.preprocessor import YoloPreprocessor


class TestYoloPreprocessor:
    """Verify resize, normalise, transpose, and batch dimension."""

    def test_output_shape_default(self) -> None:
        """Output shape is (1, 3, 640, 640) for default input_size."""
        pre = YoloPreprocessor()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.shape == (1, 3, 640, 640)

    def test_output_dtype_is_float32(self) -> None:
        pre = YoloPreprocessor()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.dtype == np.float32

    def test_output_values_in_unit_range(self) -> None:
        """All pixel values must be in [0.0, 1.0]."""
        pre = YoloPreprocessor()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_all_zeros_frame(self) -> None:
        pre = YoloPreprocessor()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.min() == 0.0
        assert result.max() == 0.0

    def test_all_white_frame(self) -> None:
        pre = YoloPreprocessor()
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = pre.process(frame)
        assert pytest.approx(result.max(), abs=1e-6) == 1.0

    def test_small_input(self) -> None:
        """Handles very small inputs (e.g., 1x1)."""
        pre = YoloPreprocessor()
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.shape == (1, 3, 640, 640)

    def test_large_input(self) -> None:
        """Handles inputs larger than the target size."""
        pre = YoloPreprocessor()
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.shape == (1, 3, 640, 640)

    def test_non_square_input(self) -> None:
        """Non-square frame is resized correctly."""
        pre = YoloPreprocessor()
        frame = np.random.randint(0, 256, (200, 800, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.shape == (1, 3, 640, 640)
        assert result.dtype == np.float32

    def test_custom_input_size(self) -> None:
        """Custom input_size produces matching output dimensions."""
        pre = YoloPreprocessor(input_size=320)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.shape == (1, 3, 320, 320)

    def test_deterministic_output(self) -> None:
        """Same input produces identical output on repeated calls."""
        pre = YoloPreprocessor()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result1 = pre.process(frame)
        result2 = pre.process(frame)
        np.testing.assert_array_equal(result1, result2)

    def test_contiguous_memory(self) -> None:
        """Output array is C-contiguous."""
        pre = YoloPreprocessor()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = pre.process(frame)
        assert result.flags["C_CONTIGUOUS"]

    def test_input_size_property(self) -> None:
        pre = YoloPreprocessor(input_size=416)
        assert pre.input_size == 416
