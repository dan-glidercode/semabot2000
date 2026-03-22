"""Fine-tune YOLO11n on a labelled dataset and export to ONNX.

Usage (on RTX PRO 6000 or similar GPU):
    python scripts/train.py \
        --data datasets/steal_a_brainrot/split/data.yaml \
        --epochs 50

    # With custom base model:
    python scripts/train.py \
        --data datasets/steal_a_brainrot/split/data.yaml \
        --model yolo11n.pt \
        --epochs 100 \
        --batch 32

After training, the ONNX model is exported to models/yolo11n_custom.onnx
and can be used by the bot with: semabot run --model models/yolo11n_custom.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def train(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    project_dir: Path,
) -> Path:
    """Run YOLO training and return path to best weights."""
    from ultralytics import YOLO

    model = YOLO(model_name)

    print(f"  Training for {epochs} epochs, batch={batch}, imgsz={imgsz}")
    print(f"  Data: {data_yaml}")
    print(f"  Output: {project_dir}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    best_pt = project_dir / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"ERROR: Expected best.pt at {best_pt}")
        sys.exit(1)

    print(f"\n  Best weights: {best_pt}")
    return best_pt


def export_onnx(weights_path: Path, imgsz: int, output_path: Path) -> Path:
    """Export trained weights to ONNX format."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    print(f"  Exporting to ONNX (imgsz={imgsz})...")

    model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)

    # ultralytics exports next to the .pt file
    exported = weights_path.with_suffix(".onnx")
    if not exported.exists():
        print(f"ERROR: Expected ONNX at {exported}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX model: {output_path} ({size_mb:.1f} MB)")
    return output_path


def validate_model(weights_path: Path, data_yaml: Path) -> None:
    """Run validation on the trained model."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    print("  Running validation...")
    results = model.val(data=str(data_yaml), verbose=False)
    print(f"  mAP50:    {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11n and export to ONNX",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml for the dataset",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base model (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, increase for 96GB VRAM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--output",
        default="models/yolo11n_custom.onnx",
        help="Output ONNX path (default: models/yolo11n_custom.onnx)",
    )
    parser.add_argument(
        "--project",
        default="runs",
        help="Training output directory (default: runs/)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export (training only)",
    )
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found: {data_yaml}")
        sys.exit(1)

    print("=" * 60)
    print("  YOLO Fine-Tuning + ONNX Export")
    print("=" * 60)

    # 1. Train
    best_pt = train(
        data_yaml,
        args.model,
        args.epochs,
        args.batch,
        args.imgsz,
        Path(args.project),
    )

    # 2. Validate
    validate_model(best_pt, data_yaml)

    # 3. Export ONNX
    if not args.skip_export:
        export_onnx(best_pt, args.imgsz, Path(args.output))

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Weights: {best_pt}")
    if not args.skip_export:
        print(f"  ONNX:    {args.output}")
    print()
    print("  To deploy, copy the ONNX file to your local machine:")
    print(f"    scp remote:{args.output} models/yolo11n_custom.onnx")
    print("  Then run: python -m semabot run --model models/yolo11n_custom.onnx")


if __name__ == "__main__":
    main()
