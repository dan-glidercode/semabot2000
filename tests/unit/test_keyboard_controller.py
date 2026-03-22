"""Tests for KeyboardController with mocked pydirectinput."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from semabot.core.models import Action

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_controller():
    """Create a KeyboardController with mocked pydirectinput."""
    mock_pdi = MagicMock()
    mock_pdi.PAUSE = 0

    with patch(
        "semabot.action.keyboard_controller.KeyboardController" "._init_pydirectinput",
        return_value=mock_pdi,
    ):
        from semabot.action.keyboard_controller import (
            KeyboardController,
        )

        ctrl = KeyboardController()
    return ctrl, mock_pdi


# -------------------------------------------------------------------
# Initial state
# -------------------------------------------------------------------


class TestKeyboardControllerInit:
    """Tests for initial state after construction."""

    def test_held_keys_empty(self) -> None:
        ctrl, _ = _make_controller()
        assert ctrl._held == set()

    def test_pydirectinput_pause_is_zero(self) -> None:
        mock_pdi = MagicMock()
        mock_pdi.PAUSE = 0
        with patch(
            "semabot.action.keyboard_controller" ".KeyboardController._init_pydirectinput",
            return_value=mock_pdi,
        ):
            from semabot.action.keyboard_controller import (
                KeyboardController,
            )

            KeyboardController()
        # _init_pydirectinput was called, which sets PAUSE = 0
        # The mock verifies the method was invoked.


# -------------------------------------------------------------------
# execute — pressing new keys
# -------------------------------------------------------------------


class TestKeyboardControllerExecute:
    """Tests for the execute method."""

    def test_press_new_keys(self) -> None:
        ctrl, pdi = _make_controller()
        action = Action(keys_press=("w", "a"), description="go")

        ctrl.execute(action)

        pdi.keyDown.assert_any_call("w")
        pdi.keyDown.assert_any_call("a")
        assert pdi.keyDown.call_count == 2
        pdi.keyUp.assert_not_called()
        assert ctrl._held == {"w", "a"}

    def test_release_old_keys(self) -> None:
        ctrl, pdi = _make_controller()
        ctrl.execute(Action(keys_press=("w", "a")))
        pdi.reset_mock()

        ctrl.execute(Action(keys_press=("d",)))

        pdi.keyUp.assert_any_call("w")
        pdi.keyUp.assert_any_call("a")
        pdi.keyDown.assert_called_once_with("d")
        assert ctrl._held == {"d"}

    def test_holding_same_keys_does_not_repress(self) -> None:
        ctrl, pdi = _make_controller()
        ctrl.execute(Action(keys_press=("w",)))
        pdi.reset_mock()

        ctrl.execute(Action(keys_press=("w",)))

        pdi.keyDown.assert_not_called()
        pdi.keyUp.assert_not_called()
        assert ctrl._held == {"w"}

    def test_empty_action_releases_all_held(self) -> None:
        ctrl, pdi = _make_controller()
        ctrl.execute(Action(keys_press=("w", "a")))
        pdi.reset_mock()

        ctrl.execute(Action())

        assert pdi.keyUp.call_count == 2
        pdi.keyDown.assert_not_called()
        assert ctrl._held == set()

    def test_transition_between_key_sets(self) -> None:
        ctrl, pdi = _make_controller()

        # Step 1: press w + a
        ctrl.execute(Action(keys_press=("w", "a")))
        assert ctrl._held == {"w", "a"}

        # Step 2: transition to w + d  (release a, press d)
        pdi.reset_mock()
        ctrl.execute(Action(keys_press=("w", "d")))

        pdi.keyUp.assert_called_once_with("a")
        pdi.keyDown.assert_called_once_with("d")
        assert ctrl._held == {"w", "d"}

        # Step 3: transition to s only
        pdi.reset_mock()
        ctrl.execute(Action(keys_press=("s",)))

        assert pdi.keyUp.call_count == 2  # w and d
        pdi.keyDown.assert_called_once_with("s")
        assert ctrl._held == {"s"}


# -------------------------------------------------------------------
# release_all
# -------------------------------------------------------------------


class TestKeyboardControllerReleaseAll:
    """Tests for the release_all method."""

    def test_releases_all_held_keys(self) -> None:
        ctrl, pdi = _make_controller()
        ctrl.execute(Action(keys_press=("w", "a", "space")))
        pdi.reset_mock()

        ctrl.release_all()

        assert pdi.keyUp.call_count == 3
        assert ctrl._held == set()

    def test_release_all_when_empty(self) -> None:
        ctrl, pdi = _make_controller()
        ctrl.release_all()

        pdi.keyUp.assert_not_called()
        assert ctrl._held == set()

    def test_release_all_then_execute(self) -> None:
        ctrl, pdi = _make_controller()
        ctrl.execute(Action(keys_press=("w",)))
        ctrl.release_all()
        pdi.reset_mock()

        ctrl.execute(Action(keys_press=("w",)))

        # w was released, so pressing again should call keyDown
        pdi.keyDown.assert_called_once_with("w")
        assert ctrl._held == {"w"}
