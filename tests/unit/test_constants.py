"""Tests for core.constants."""

from __future__ import annotations

from semabot.core.constants import COCO_CLASSES, DEFAULT_CONFIG_PATH, DEFAULT_GAMES_DIR


def test_coco_classes_has_80_entries():
    assert len(COCO_CLASSES) == 80


def test_coco_classes_first_is_person():
    assert COCO_CLASSES[0] == "person"


def test_coco_classes_is_tuple():
    assert isinstance(COCO_CLASSES, tuple)


def test_default_config_path():
    assert DEFAULT_CONFIG_PATH == "config/default.toml"


def test_default_games_dir():
    assert DEFAULT_GAMES_DIR == "config/games"
