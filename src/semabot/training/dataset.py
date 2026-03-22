"""Dataset utilities — split, YAML generation, and validation."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def split_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, int]:
    """Split images and labels into train/val directories.

    Parameters
    ----------
    images_dir:
        Directory containing ``.png`` source images.
    labels_dir:
        Directory containing YOLO-format ``.txt`` label files.
    output_dir:
        Root of the split dataset.  Sub-directories
        ``images/train``, ``images/val``, ``labels/train``,
        ``labels/val`` are created automatically.
    train_ratio:
        Fraction of images assigned to the training split.
    seed:
        Random seed for reproducible shuffling.

    Returns
    -------
    dict
        ``{"train": <count>, "val": <count>}``
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    out = Path(output_dir)

    image_files = sorted(images_path.glob("*.png"))
    random.seed(seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_ratio)
    splits: dict[str, list[Path]] = {
        "train": image_files[:split_idx],
        "val": image_files[split_idx:],
    }

    for split_name, files in splits.items():
        _copy_split(
            files,
            labels_path,
            out,
            split_name,
        )

    result = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
    }
    logger.info(
        "Split complete: %d train, %d val",
        result["train"],
        result["val"],
    )
    return result


def generate_data_yaml(
    output_dir: str | Path,
    class_names: list[str],
) -> Path:
    """Write a ``data.yaml`` for YOLO training.

    Parameters
    ----------
    output_dir:
        Root dataset directory (must contain ``images/train``
        and ``images/val``).
    class_names:
        Ordered list of class names.

    Returns
    -------
    Path
        Absolute path to the written ``data.yaml``.
    """
    out = Path(output_dir)
    yaml_path = out / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"train: {out / 'images' / 'train'}\n")
        fh.write(f"val: {out / 'images' / 'val'}\n")
        fh.write(f"nc: {len(class_names)}\n")
        fh.write(f"names: {class_names}\n")

    logger.info("Wrote data.yaml to %s", yaml_path)
    return yaml_path


def validate_dataset(
    dataset_dir: str | Path,
) -> dict[str, object]:
    """Validate a split dataset directory.

    Checks that the expected sub-directories and ``data.yaml``
    exist, and counts images and labels in each split.

    Returns
    -------
    dict
        Keys: ``valid``, ``train_images``, ``val_images``,
        ``train_labels``, ``val_labels``, ``errors``.
    """
    root = Path(dataset_dir)
    errors: list[str] = []

    required_dirs = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    for d in required_dirs:
        if not d.is_dir():
            errors.append(f"Missing directory: {d}")

    if not (root / "data.yaml").is_file():
        errors.append("Missing data.yaml")

    counts = _count_files(root)
    return {
        "valid": len(errors) == 0,
        "train_images": counts["train_images"],
        "val_images": counts["val_images"],
        "train_labels": counts["train_labels"],
        "val_labels": counts["val_labels"],
        "errors": errors,
    }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _copy_split(
    image_files: list[Path],
    labels_path: Path,
    output_dir: Path,
    split_name: str,
) -> None:
    """Copy images and labels into a split sub-directory."""
    img_dest = output_dir / "images" / split_name
    lbl_dest = output_dir / "labels" / split_name
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    for img_file in image_files:
        shutil.copy2(img_file, img_dest / img_file.name)
        label_file = labels_path / img_file.with_suffix(".txt").name
        if label_file.exists():
            shutil.copy2(label_file, lbl_dest / label_file.name)
        else:
            _create_empty_label(lbl_dest / label_file.name)


def _create_empty_label(path: Path) -> None:
    """Create an empty label file at *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("")


def _count_files(root: Path) -> dict[str, int]:
    """Count ``.png`` images and ``.txt`` labels per split."""
    return {
        "train_images": _count_glob(
            root / "images" / "train",
            "*.png",
        ),
        "val_images": _count_glob(
            root / "images" / "val",
            "*.png",
        ),
        "train_labels": _count_glob(
            root / "labels" / "train",
            "*.txt",
        ),
        "val_labels": _count_glob(
            root / "labels" / "val",
            "*.txt",
        ),
    }


def _count_glob(directory: Path, pattern: str) -> int:
    """Return count of files matching *pattern* in *directory*."""
    if not directory.is_dir():
        return 0
    return len(list(directory.glob(pattern)))
