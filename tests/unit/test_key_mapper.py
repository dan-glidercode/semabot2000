"""Tests for KeyMapper."""

from __future__ import annotations

import pytest

from semabot.action.key_mapper import KeyMapper
from semabot.core.config import GameControls

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


def _default_controls() -> GameControls:
    return GameControls(
        move_forward="w",
        move_backward="s",
        move_left="a",
        move_right="d",
        camera_left="q",
        camera_right="e",
        jump="space",
        interact="f",
    )


# -------------------------------------------------------------------
# get_key
# -------------------------------------------------------------------


class TestKeyMapperGetKey:
    """Tests for single-key mapping."""

    def test_maps_move_forward(self) -> None:
        mapper = KeyMapper(_default_controls())
        assert mapper.get_key("move_forward") == "w"

    def test_maps_jump(self) -> None:
        mapper = KeyMapper(_default_controls())
        assert mapper.get_key("jump") == "space"

    def test_maps_interact(self) -> None:
        mapper = KeyMapper(_default_controls())
        assert mapper.get_key("interact") == "f"

    def test_all_controls_mapped(self) -> None:
        mapper = KeyMapper(_default_controls())
        expected = {
            "move_forward": "w",
            "move_backward": "s",
            "move_left": "a",
            "move_right": "d",
            "camera_left": "q",
            "camera_right": "e",
            "jump": "space",
            "interact": "f",
        }
        for name, key in expected.items():
            assert mapper.get_key(name) == key

    def test_unknown_control_raises_key_error(self) -> None:
        mapper = KeyMapper(_default_controls())
        with pytest.raises(KeyError, match="nonexistent"):
            mapper.get_key("nonexistent")


# -------------------------------------------------------------------
# get_keys
# -------------------------------------------------------------------


class TestKeyMapperGetKeys:
    """Tests for multi-key mapping."""

    def test_maps_multiple(self) -> None:
        mapper = KeyMapper(_default_controls())
        result = mapper.get_keys(["move_forward", "move_left", "jump"])
        assert result == ["w", "a", "space"]

    def test_empty_list(self) -> None:
        mapper = KeyMapper(_default_controls())
        assert mapper.get_keys([]) == []

    def test_single_item(self) -> None:
        mapper = KeyMapper(_default_controls())
        assert mapper.get_keys(["move_right"]) == ["d"]

    def test_preserves_order(self) -> None:
        mapper = KeyMapper(_default_controls())
        result = mapper.get_keys(["interact", "jump", "move_backward"])
        assert result == ["f", "space", "s"]


# -------------------------------------------------------------------
# validate
# -------------------------------------------------------------------


class TestKeyMapperValidate:
    """Tests for validate method."""

    def test_valid_controls_pass(self) -> None:
        mapper = KeyMapper(_default_controls())
        mapper.validate()  # should not raise

    def test_custom_key_bindings_pass(self) -> None:
        controls = GameControls(
            move_forward="up",
            move_backward="down",
            move_left="left",
            move_right="right",
            camera_left="z",
            camera_right="x",
            jump="ctrl",
            interact="e",
        )
        mapper = KeyMapper(controls)
        mapper.validate()  # should not raise
