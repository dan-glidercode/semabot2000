"""CLI entry point — ``python -m semabot <command> [options]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from semabot.core.constants import DEFAULT_CONFIG_PATH, DEFAULT_GAMES_DIR

logger = logging.getLogger(__name__)

DEFAULT_CAPTURE_OUTPUT = "output/captures"
DEFAULT_MODEL_OUTPUT = "models/yolo11n.onnx"
DEFAULT_INPUT_SIZE = 640


# -------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse parser and return parsed arguments."""
    parser = argparse.ArgumentParser(
        prog="semabot",
        description="SeMaBot2000 — AI-powered Roblox bot",
    )
    sub = parser.add_subparsers(dest="command")

    _add_run_parser(sub)
    _add_capture_parser(sub)
    _add_detect_parser(sub)
    _add_export_model_parser(sub)

    return parser.parse_args(argv)


def _add_run_parser(
    sub: argparse._SubParsersAction,
) -> None:
    """Register the ``run`` subcommand."""
    p = sub.add_parser("run", help="Run the bot")
    p.add_argument("--game", required=True, help="Game profile name")
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Bot config path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log actions without sending input",
    )
    p.add_argument(
        "--save-detections",
        action="store_true",
        default=False,
        help="Save annotated frames",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds then stop",
    )
    p.add_argument(
        "--log-level",
        default=None,
        help="Override log level",
    )


def _add_capture_parser(
    sub: argparse._SubParsersAction,
) -> None:
    """Register the ``capture`` subcommand."""
    p = sub.add_parser("capture", help="Capture test frames")
    p.add_argument(
        "--method",
        default="wgc",
        help="Capture method: wgc or mss",
    )
    p.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of frames to capture",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_CAPTURE_OUTPUT,
        help="Output directory",
    )


def _add_detect_parser(
    sub: argparse._SubParsersAction,
) -> None:
    """Register the ``detect`` subcommand."""
    p = sub.add_parser(
        "detect",
        help="Run detection on a static image",
    )
    p.add_argument("image_path", help="Path to the image file")
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Bot config path",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save annotated image to this path",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override confidence threshold",
    )


def _add_export_model_parser(
    sub: argparse._SubParsersAction,
) -> None:
    """Register the ``export-model`` subcommand."""
    p = sub.add_parser(
        "export-model",
        help="Export YOLO weights to ONNX",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_MODEL_OUTPUT,
        help="Output ONNX path",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Export input size",
    )


# -------------------------------------------------------------------
# Subcommand handlers
# -------------------------------------------------------------------


def _handle_run(args: argparse.Namespace) -> None:
    """Handle the ``run`` subcommand."""
    from semabot.app.factory import create_bot

    game_profile_path = Path(DEFAULT_GAMES_DIR) / f"{args.game}.toml"
    orchestrator = create_bot(
        config_path=args.config,
        game_profile_path=game_profile_path,
        dry_run=args.dry_run,
    )
    logger.info("Starting bot for game '%s'", args.game)
    orchestrator.run(duration=args.duration)


def _handle_capture(args: argparse.Namespace) -> None:
    """Handle the ``capture`` subcommand."""

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    method = args.method.lower()

    frame_source = _make_capture_source(method)
    frame_source.start()

    try:
        _capture_frames(frame_source, args.count, output_dir)
    finally:
        frame_source.stop()


def _make_capture_source(method: str) -> object:
    """Build a frame source for the capture subcommand."""
    if method == "wgc":
        from semabot.capture.wgc_source import WGCFrameSource

        return WGCFrameSource(window_title="Roblox")
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


def _capture_frames(
    frame_source: object,
    count: int,
    output_dir: Path,
) -> None:
    """Grab *count* frames and save them as PNGs."""
    import time

    import cv2

    saved = 0
    attempts = 0
    max_attempts = count * 20

    while saved < count and attempts < max_attempts:
        attempts += 1
        frame = frame_source.get_latest_frame()  # type: ignore[union-attr]
        if frame is None:
            time.sleep(0.05)
            continue
        filename = output_dir / f"frame_{saved:04d}.png"
        cv2.imwrite(str(filename), frame)
        logger.info("Saved %s", filename)
        saved += 1
        time.sleep(0.1)

    logger.info("Captured %d / %d frames", saved, count)


def _handle_detect(args: argparse.Namespace) -> None:
    """Handle the ``detect`` subcommand."""
    import cv2

    from semabot.core.config import load_config
    from semabot.intelligence.detector import YoloDetector
    from semabot.intelligence.preprocessor import YoloPreprocessor

    config = load_config(args.config)
    threshold = (
        args.threshold if args.threshold is not None else config.detection.confidence_threshold
    )

    image = cv2.imread(args.image_path)
    if image is None:
        logger.error("Cannot read image: %s", args.image_path)
        sys.exit(1)

    preprocessor = YoloPreprocessor(config.detection.input_size)
    detector = YoloDetector(
        model_path=config.detection.model_path,
        provider=config.detection.provider,
        confidence_threshold=threshold,
        nms_iou_threshold=config.detection.nms_iou_threshold,
    )

    blob = preprocessor.process(image)
    detections = detector.detect(blob)

    for det in detections:
        print(
            f"{det.class_name}: {det.confidence:.2f} "
            f"[{det.bbox.x1:.0f}, {det.bbox.y1:.0f}, "
            f"{det.bbox.x2:.0f}, {det.bbox.y2:.0f}]"
        )

    if args.output is not None:
        _save_annotated(image, detections, args.output)


def _save_annotated(
    image: object,
    detections: list,
    output_path: str,
) -> None:
    """Draw bounding boxes on *image* and save to *output_path*."""
    import cv2

    for det in detections:
        x1 = int(det.bbox.x1)
        y1 = int(det.bbox.y1)
        x2 = int(det.bbox.x2)
        y2 = int(det.bbox.y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(output_path, image)
    logger.info("Annotated image saved to %s", output_path)


def _handle_export_model(args: argparse.Namespace) -> None:
    """Handle the ``export-model`` subcommand."""
    print("Export-model is not yet implemented.")
    print(f"  Output path: {args.output}")
    print(f"  Input size:  {args.input_size}")
    print()
    print("To export manually, run:")
    print("  pip install ultralytics")
    print("  yolo export model=yolo11n.pt format=onnx " f"imgsz={args.input_size}")
    print(f"  mv yolo11n.onnx {args.output}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate handler."""
    args = parse_args(argv)

    log_level = getattr(args, "log_level", None) or "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    handlers = {
        "run": _handle_run,
        "capture": _handle_capture,
        "detect": _handle_detect,
        "export-model": _handle_export_model,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print("Usage: semabot <command> [options]")
        print("Commands: run, capture, detect, export-model")
        sys.exit(1)

    handler(args)
