"""Tests for core.protocols — verify protocol classes are importable and structurally correct."""

from __future__ import annotations

import typing

from semabot.core.protocols import (
    DecisionEngine,
    Detector,
    FrameSource,
    InputController,
    Preprocessor,
    StateBuilder,
)


def test_all_protocols_are_protocol_subclasses():
    for proto in [
        FrameSource,
        Preprocessor,
        Detector,
        StateBuilder,
        DecisionEngine,
        InputController,
    ]:
        assert issubclass(proto, typing.Protocol), f"{proto.__name__} is not a Protocol"


def test_frame_source_has_required_methods():
    methods = {"start", "get_latest_frame", "stop"}
    actual = {m for m in dir(FrameSource) if not m.startswith("_")}
    assert methods.issubset(actual)


def test_preprocessor_has_process():
    assert hasattr(Preprocessor, "process")


def test_detector_has_detect():
    assert hasattr(Detector, "detect")


def test_state_builder_has_build():
    assert hasattr(StateBuilder, "build")


def test_decision_engine_has_decide():
    assert hasattr(DecisionEngine, "decide")


def test_input_controller_has_execute_and_release_all():
    assert hasattr(InputController, "execute")
    assert hasattr(InputController, "release_all")
