"""Auto-label game frames using Grounding DINO via autodistill.

Reads an ontology.json (class_name -> text_prompt) and uses Grounding
DINO to generate YOLO-format bounding box labels for all frames.

Usage (on RTX PRO 6000 or similar GPU):
    python scripts/autodistill_label.py \
        --ontology config/ontologies/steal_a_brainrot.json \
        --images datasets/steal_a_brainrot/images \
        --output datasets/steal_a_brainrot/labels

Requires: pip install autodistill autodistill-grounding-dino
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def load_ontology(path: Path) -> dict[str, str]:
    """Load class_name -> prompt mapping from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        print(f"ERROR: Empty ontology at {path}")
        sys.exit(1)
    return data


def run_labeling(
    ontology_map: dict[str, str],
    images_dir: Path,
    output_dir: Path,
    threshold: float,
) -> None:
    """Run Grounding DINO on all frames and write YOLO labels."""
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO

    # CaptionOntology maps prompt -> class_name
    caption_map = {prompt: cls for cls, prompt in ontology_map.items()}
    ontology = CaptionOntology(caption_map)

    model = GroundingDINO(
        ontology=ontology,
        box_threshold=threshold,
        text_threshold=threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"ERROR: No .png files found in {images_dir}")
        sys.exit(1)

    class_names = list(ontology_map.keys())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    print(f"  Labeling {len(image_files)} images with {len(class_names)} classes...")
    print(f"  Classes: {class_names}")
    print(f"  Threshold: {threshold}")

    total_detections = 0
    t_start = time.perf_counter()

    for i, img_path in enumerate(image_files):
        detections = model.predict(str(img_path))

        label_path = output_dir / img_path.with_suffix(".txt").name
        lines = _detections_to_yolo(
            detections, class_names, class_to_id, img_path,
        )
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        total_detections += len(lines)
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(image_files)}] "
                  f"{total_detections} detections, {fps:.1f} img/s")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Done: {len(image_files)} images, "
          f"{total_detections} detections, {elapsed:.1f}s")

    # Write class names file for reference
    names_path = output_dir / "classes.txt"
    with open(names_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"  Class list saved to {names_path}")


def _detections_to_yolo(
    detections: object,
    class_names: list[str],
    class_to_id: dict[str, int],
    img_path: Path,
) -> list[str]:
    """Convert autodistill detections to YOLO-format lines."""
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return []
    img_h, img_w = img.shape[:2]

    lines: list[str] = []
    # autodistill returns sv.Detections with xyxy, confidence, class_id
    for j in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[j]
        cls_idx = int(detections.class_id[j])

        if cls_idx >= len(class_names):
            continue

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label frames with Grounding DINO",
    )
    parser.add_argument(
        "--ontology",
        required=True,
        help="Path to ontology.json (class -> prompt mapping)",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to directory containing .png frames",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for .txt labels (default: <images>/../labels)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    args = parser.parse_args()

    ontology_path = Path(args.ontology)
    images_dir = Path(args.images)
    output_dir = Path(args.output) if args.output else images_dir.parent / "labels"

    if not ontology_path.exists():
        print(f"ERROR: Ontology not found: {ontology_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"ERROR: Images dir not found: {images_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Grounding DINO Auto-Labeling")
    print("=" * 60)
    print(f"  Ontology: {ontology_path}")
    print(f"  Images:   {images_dir}")
    print(f"  Output:   {output_dir}")

    ontology_map = load_ontology(ontology_path)
    print(f"  Classes:  {len(ontology_map)}")
    for cls, prompt in ontology_map.items():
        print(f"    {cls:30s} -> {prompt}")

    run_labeling(ontology_map, images_dir, output_dir, args.threshold)


if __name__ == "__main__":
    main()
