"""Component factory — assembles a BotOrchestrator from config files."""

from __future__ import annotations

from pathlib import Path

from semabot.app.orchestrator import BotOrchestrator
from semabot.core.config import (
    BotConfig,
    GameProfile,
    load_config,
    load_game_profile,
)


def _create_frame_source(
    config: BotConfig,
) -> object:
    """Instantiate the frame source specified by *config*."""
    method = config.capture.method.lower()
    if method == "wgc":
        from semabot.capture.wgc_source import WGCFrameSource

        return WGCFrameSource(
            window_title=config.capture.window_title,
        )
    if method == "mss":
        from semabot.capture.mss_source import MSSFrameSource

        return MSSFrameSource(
            window_region={
                "left": 0,
                "top": 0,
                "width": 1920,
                "height": 1080,
            },
        )
    msg = f"Unknown capture method: '{method}'"
    raise ValueError(msg)


def _create_decision_engine(
    profile: GameProfile,
) -> object:
    """Build the behavior tree engine for *profile*."""
    from semabot.intelligence.behavior.engine import (
        BehaviorTreeEngine,
    )
    from semabot.intelligence.behavior.trees import (
        build_steal_a_brainrot_tree,
    )

    tree = build_steal_a_brainrot_tree(profile)
    return BehaviorTreeEngine(tree)


def _create_input_controller(dry_run: bool) -> object:
    """Return a real or null controller based on *dry_run*."""
    if dry_run:
        from semabot.action.null_controller import (
            NullInputController,
        )

        return NullInputController()

    from semabot.action.keyboard_controller import (
        KeyboardController,
    )

    return KeyboardController()


def create_bot(
    config_path: str | Path,
    game_profile_path: str | Path,
    dry_run: bool = False,
) -> BotOrchestrator:
    """Assemble a fully-wired :class:`BotOrchestrator`.

    Parameters
    ----------
    config_path:
        Path to the bot TOML configuration file.
    game_profile_path:
        Path to the game profile TOML file.
    dry_run:
        When *True*, use a :class:`NullInputController` that
        logs actions instead of sending real key presses.

    Returns
    -------
    BotOrchestrator
        Ready-to-run orchestrator with all components wired.
    """
    config: BotConfig = load_config(config_path)
    profile: GameProfile = load_game_profile(game_profile_path)

    from semabot.intelligence.detector import YoloDetector
    from semabot.intelligence.preprocessor import YoloPreprocessor
    from semabot.intelligence.state_builder import GameStateBuilder

    frame_source = _create_frame_source(config)
    preprocessor = YoloPreprocessor(config.detection.input_size)
    detector = YoloDetector(
        model_path=config.detection.model_path,
        provider=config.detection.provider,
        confidence_threshold=config.detection.confidence_threshold,
        nms_iou_threshold=config.detection.nms_iou_threshold,
    )
    state_builder = GameStateBuilder(
        config.detection.confidence_threshold,
    )
    decision_engine = _create_decision_engine(profile)
    input_controller = _create_input_controller(dry_run)

    return BotOrchestrator(
        frame_source=frame_source,
        preprocessor=preprocessor,
        detector=detector,
        state_builder=state_builder,
        decision_engine=decision_engine,
        input_controller=input_controller,
        config=config,
    )
