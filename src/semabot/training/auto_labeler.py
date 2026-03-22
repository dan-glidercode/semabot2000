"""Auto-labeler — generates YOLO-format labels from detections."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoLabeler:
    """Generates YOLO-format label files using a detector.

    Parameters
    ----------
    detector:
        Any object with a ``detect(blob)`` method returning a list
        of Detection objects.
    preprocessor:
        Any object with a ``process(frame)`` method that returns
        a preprocessed blob.
    class_map:
        Optional mapping of class names to integer IDs.  When
        ``None``, IDs are assigned sequentially as new classes
        are encountered.
    """

    def __init__(
        self,
        detector: object,
        preprocessor: object,
        class_map: dict[str, int] | None = None,
    ) -> None:
        self._detector = detector
        self._preprocessor = preprocessor
        self._class_map: dict[str, int] = dict(class_map) if class_map else {}
        self._auto_id = max(self._class_map.values()) + 1 if self._class_map else 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_dataset(
        self,
        images_dir: str,
        output_dir: str | None = None,
        threshold: float = 0.3,
    ) -> int:
        """Label every ``.png`` image in *images_dir*.

        Returns the number of images labelled.
        """
        import cv2

        imgs_path = Path(images_dir)
        labels_path = self._resolve_output_dir(imgs_path, output_dir)
        labels_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(imgs_path.glob("*.png"))
        labelled = 0

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            self._label_single_image(image, img_file, labels_path, threshold)
            labelled += 1
            if labelled % 10 == 0:
                logger.info("Labelled %d images", labelled)

        self._write_data_yaml(labels_path, imgs_path, image_files)
        logger.info("Labelling complete: %d images", labelled)
        return labelled

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_output_dir(
        imgs_path: Path,
        output_dir: str | None,
    ) -> Path:
        """Return the labels directory path."""
        if output_dir is not None:
            return Path(output_dir)
        return imgs_path.parent / "labels"

    def _label_single_image(
        self,
        image: object,
        img_file: Path,
        labels_path: Path,
        threshold: float,
    ) -> None:
        """Detect, filter, and write a YOLO label file."""
        blob = self._preprocessor.process(image)  # type: ignore[union-attr]
        detections = self._detector.detect(blob)  # type: ignore[union-attr]
        input_h = blob.shape[2]  # type: ignore[union-attr]
        input_w = blob.shape[3]  # type: ignore[union-attr]

        label_file = labels_path / img_file.with_suffix(".txt").name
        lines = self._build_label_lines(detections, threshold, input_w, input_h)
        with open(label_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def _build_label_lines(
        self,
        detections: list,
        threshold: float,
        input_w: int,
        input_h: int,
    ) -> list[str]:
        """Build YOLO-format label lines for filtered detections."""
        lines: list[str] = []
        for det in detections:
            if det.confidence < threshold:
                continue
            class_id = self._get_class_id(det.class_name)
            cx, cy = det.bbox.center
            w = det.bbox.width
            h = det.bbox.height
            lines.append(
                f"{class_id} "
                f"{cx / input_w:.6f} "
                f"{cy / input_h:.6f} "
                f"{w / input_w:.6f} "
                f"{h / input_h:.6f}"
            )
        return lines

    def _get_class_id(self, class_name: str) -> int:
        """Return (or assign) the integer ID for *class_name*."""
        if class_name not in self._class_map:
            self._class_map[class_name] = self._auto_id
            self._auto_id += 1
        return self._class_map[class_name]

    def _write_data_yaml(
        self,
        labels_path: Path,
        imgs_path: Path,
        image_files: list[Path],
    ) -> None:
        """Write a ``data.yaml`` alongside the labels directory."""
        if not image_files:
            return
        names = {v: k for k, v in self._class_map.items()}
        sorted_ids = sorted(names.keys())
        names_list = [names[i] for i in sorted_ids]

        yaml_path = labels_path.parent / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write(f"train: {imgs_path}\n")
            fh.write(f"val: {imgs_path}\n")
            fh.write(f"nc: {len(names_list)}\n")
            fh.write(f"names: {names_list}\n")
