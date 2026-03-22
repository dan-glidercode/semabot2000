"""YOLO image preprocessor — resize, normalize, transpose."""

from __future__ import annotations

import cv2
import numpy as np


class YoloPreprocessor:
    """Implements the Preprocessor protocol for YOLO11n.

    Resizes a raw BGR frame to *input_size* x *input_size*, converts
    HWC -> CHW layout, normalises pixel values to [0, 1] float32, and
    adds a batch dimension so the result is ready for ONNX inference.
    """

    def __init__(self, input_size: int = 640) -> None:
        self._input_size = input_size

    @property
    def input_size(self) -> int:
        """Return the configured square input dimension."""
        return self._input_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise, and reshape *frame* for the detector.

        Parameters
        ----------
        frame:
            BGR image with shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Float32 tensor with shape ``(1, 3, input_size, input_size)``
            and values in ``[0.0, 1.0]``.
        """
        resized = cv2.resize(
            frame,
            (self._input_size, self._input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        blob: np.ndarray = np.ascontiguousarray(
            resized.transpose(2, 0, 1),
            dtype=np.float32,
        )
        blob = blob * (1.0 / 255.0)
        return blob[np.newaxis]
