"""YOLO11n ONNX detector — inference, decode, NMS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from semabot.core.constants import COCO_CLASSES
from semabot.core.models import BoundingBox, Detection

if TYPE_CHECKING:
    import onnxruntime as ort


class YoloDetector:
    """Implements the Detector protocol using YOLO11n via ONNX Runtime.

    The ONNX session is lazily created on the first call to
    :meth:`detect` so that unit tests can monkey-patch
    ``_session`` without needing a real model file.
    """

    def __init__(
        self,
        model_path: str,
        provider: str = "DmlExecutionProvider",
        confidence_threshold: float = 0.35,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        self._model_path = model_path
        self._provider = provider
        self._confidence_threshold = confidence_threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._session: ort.InferenceSession | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, blob: np.ndarray) -> list[Detection]:
        """Run inference on *blob* and return filtered detections.

        Parameters
        ----------
        blob:
            Float32 tensor of shape ``(1, 3, H, W)``.

        Returns
        -------
        list[Detection]
            Post-processed detections after confidence filtering
            and NMS.
        """
        self._ensure_session()
        assert self._session is not None  # for type-checker

        input_name = self._session.get_inputs()[0].name
        raw_output: np.ndarray = self._session.run(
            None,
            {input_name: blob},
        )[0]

        return self._decode_output(raw_output)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_session(self) -> None:
        """Create the ONNX inference session if it does not exist."""
        if self._session is not None:
            return
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            self._model_path,
            providers=[self._provider],
        )

    def _decode_output(
        self,
        raw: np.ndarray,
    ) -> list[Detection]:
        """Decode YOLO11n output into a list of :class:`Detection`.

        The raw output has shape ``(1, 84, 8400)``.  After
        transposing to ``(8400, 84)`` each row contains:

        * cols 0-3: ``cx, cy, w, h`` (centre-format bounding box)
        * cols 4-83: class confidence scores (80 COCO classes)
        """
        # (1, 84, 8400) -> (8400, 84)
        predictions = raw[0].T

        # Bounding boxes: centre format -> corner format
        cx = predictions[:, 0]
        cy = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Class scores
        scores_all = predictions[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(len(scores_all)), class_ids]

        # Confidence filter
        mask = confidences >= self._confidence_threshold
        x1 = x1[mask]
        y1 = y1[mask]
        x2 = x2[mask]
        y2 = y2[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return []

        # NMS
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._apply_nms(boxes, confidences)

        detections: list[Detection] = []
        for idx in keep:
            cid = int(class_ids[idx])
            class_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(confidences[idx]),
                    bbox=BoundingBox(
                        x1=float(x1[idx]),
                        y1=float(y1[idx]),
                        x2=float(x2[idx]),
                        y2=float(y2[idx]),
                    ),
                )
            )
        return detections

    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        """Greedy non-maximum suppression.

        Parameters
        ----------
        boxes:
            ``(N, 4)`` array of ``[x1, y1, x2, y2]`` boxes.
        scores:
            ``(N,)`` confidence scores.

        Returns
        -------
        list[int]
            Indices of boxes to keep.
        """
        order = scores.argsort()[::-1].tolist()
        keep: list[int] = []

        while order:
            i = order.pop(0)
            keep.append(i)
            remaining: list[int] = []
            for j in order:
                if self._iou(boxes[i], boxes[j]) < self._nms_iou_threshold:
                    remaining.append(j)
            order = remaining

        return keep

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """Compute IoU between two ``[x1, y1, x2, y2]`` boxes."""
        inter_x1 = max(a[0], b[0])
        inter_y1 = max(a[1], b[1])
        inter_x2 = min(a[2], b[2])
        inter_y2 = min(a[3], b[3])

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0
        return float(inter_area / union)
