"""Tests for dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from semabot.training.dataset import (
    generate_data_yaml,
    split_dataset,
    validate_dataset,
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _create_test_images(
    images_dir: Path,
    count: int = 10,
) -> list[str]:
    """Write tiny PNGs and return their stem names."""
    import cv2

    images_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i in range(1, count + 1):
        name = f"frame_{i:06d}"
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"{name}.png"), img)
        names.append(name)
    return names


def _create_test_labels(
    labels_dir: Path,
    names: list[str],
    skip: list[str] | None = None,
) -> None:
    """Write one-line YOLO label files, skipping *skip*."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    skip = skip or []
    for name in names:
        if name in skip:
            continue
        label = labels_dir / f"{name}.txt"
        label.write_text("0 0.5 0.5 0.1 0.1\n")


def _build_valid_dataset(
    root: Path,
    class_names: list[str],
    image_count: int = 10,
) -> None:
    """Create a fully valid split dataset at *root*."""
    images_dir = root / "source_images"
    labels_dir = root / "source_labels"
    names = _create_test_images(images_dir, count=image_count)
    _create_test_labels(labels_dir, names)
    split_dataset(images_dir, labels_dir, root)
    generate_data_yaml(root, class_names)


# -------------------------------------------------------------------
# Tests — split_dataset
# -------------------------------------------------------------------


class TestSplitDatasetCounts:
    """Verify train/val counts with default 80/20 split."""

    def test_80_20_split(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        names = _create_test_images(images_dir, count=10)
        _create_test_labels(labels_dir, names)

        out = tmp_path / "output"
        result = split_dataset(images_dir, labels_dir, out)

        assert result == {"train": 8, "val": 2}

    def test_files_exist_in_correct_dirs(
        self,
        tmp_path: Path,
    ) -> None:
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        names = _create_test_images(images_dir, count=10)
        _create_test_labels(labels_dir, names)

        out = tmp_path / "output"
        split_dataset(images_dir, labels_dir, out)

        train_imgs = list((out / "images" / "train").glob("*.png"))
        val_imgs = list((out / "images" / "val").glob("*.png"))
        assert len(train_imgs) == 8
        assert len(val_imgs) == 2

    def test_label_files_present(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        names = _create_test_images(images_dir, count=10)
        _create_test_labels(labels_dir, names)

        out = tmp_path / "output"
        split_dataset(images_dir, labels_dir, out)

        train_labels = list(
            (out / "labels" / "train").glob("*.txt"),
        )
        val_labels = list(
            (out / "labels" / "val").glob("*.txt"),
        )
        assert len(train_labels) == 8
        assert len(val_labels) == 2


class TestSplitDatasetMissingLabels:
    """Verify empty .txt is created for missing labels."""

    def test_missing_label_creates_empty_txt(
        self,
        tmp_path: Path,
    ) -> None:
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        names = _create_test_images(images_dir, count=5)
        # Skip labels for the first two images
        _create_test_labels(labels_dir, names, skip=names[:2])

        out = tmp_path / "output"
        split_dataset(images_dir, labels_dir, out)

        all_labels = list(
            (out / "labels" / "train").glob("*.txt"),
        ) + list(
            (out / "labels" / "val").glob("*.txt"),
        )
        assert len(all_labels) == 5

        # At least one of the missing-label files should be empty
        empty_files = [lf for lf in all_labels if lf.read_text().strip() == ""]
        assert len(empty_files) >= 1


class TestSplitDatasetCustomRatio:
    """Verify custom train_ratio works."""

    def test_50_50_split(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        names = _create_test_images(images_dir, count=10)
        _create_test_labels(labels_dir, names)

        out = tmp_path / "output"
        result = split_dataset(
            images_dir,
            labels_dir,
            out,
            train_ratio=0.5,
        )

        assert result == {"train": 5, "val": 5}

    def test_90_10_split(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        names = _create_test_images(images_dir, count=20)
        _create_test_labels(labels_dir, names)

        out = tmp_path / "output"
        result = split_dataset(
            images_dir,
            labels_dir,
            out,
            train_ratio=0.9,
        )

        assert result == {"train": 18, "val": 2}


# -------------------------------------------------------------------
# Tests — generate_data_yaml
# -------------------------------------------------------------------


class TestGenerateDataYaml:
    """Verify data.yaml content and structure."""

    def test_yaml_content(self, tmp_path: Path) -> None:
        class_names = ["player", "enemy", "coin"]
        yaml_path = generate_data_yaml(tmp_path, class_names)

        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "train:" in content
        assert "val:" in content
        assert "nc: 3" in content
        assert "player" in content
        assert "enemy" in content
        assert "coin" in content

    def test_nc_matches_class_count(
        self,
        tmp_path: Path,
    ) -> None:
        class_names = ["a", "b"]
        yaml_path = generate_data_yaml(tmp_path, class_names)

        content = yaml_path.read_text()
        assert "nc: 2" in content

    def test_returns_path(self, tmp_path: Path) -> None:
        yaml_path = generate_data_yaml(tmp_path, ["x"])

        assert isinstance(yaml_path, Path)
        assert yaml_path.name == "data.yaml"


# -------------------------------------------------------------------
# Tests — validate_dataset
# -------------------------------------------------------------------


class TestValidateDatasetValid:
    """Verify a correctly structured dataset passes."""

    def test_valid_dataset(self, tmp_path: Path) -> None:
        _build_valid_dataset(tmp_path, ["player", "enemy"])

        result = validate_dataset(tmp_path)

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["train_images"] == 8
        assert result["val_images"] == 2
        assert result["train_labels"] == 8
        assert result["val_labels"] == 2


class TestValidateDatasetMissingDirs:
    """Verify missing directories are reported."""

    def test_empty_dir_reports_errors(
        self,
        tmp_path: Path,
    ) -> None:
        result = validate_dataset(tmp_path)

        assert result["valid"] is False
        assert len(result["errors"]) >= 4  # 4 dirs + data.yaml

    def test_missing_val_dir(self, tmp_path: Path) -> None:
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)
        (tmp_path / "data.yaml").write_text("nc: 1\n")

        result = validate_dataset(tmp_path)

        assert result["valid"] is False
        errors_text = " ".join(result["errors"])
        assert "images" in errors_text and "val" in errors_text


class TestValidateDatasetMissingYaml:
    """Verify missing data.yaml is reported."""

    def test_no_data_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)

        result = validate_dataset(tmp_path)

        assert result["valid"] is False
        assert any("data.yaml" in e for e in result["errors"])
