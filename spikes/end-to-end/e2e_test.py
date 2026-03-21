"""
End-to-End Pipeline Spike — SeMaBot2000
=========================================
Full pipeline: WGC capture -> preprocess -> YOLO11n -> postprocess -> BT decide -> act

The bot will chase the nearest detected "person" for 3 seconds using
keyboard-only controls (WASD + arrow keys), then stop and report metrics.

Prerequisites:
  - Roblox running, joined a game, window visible
  - Run from a regular terminal (not VS Code embedded)

Run: python e2e_test.py
"""

import sys
import time
import threading
import statistics
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import cv2
import onnxruntime as ort
import py_trees
import pydirectinput
import win32gui

from windows_capture import WindowsCapture, Frame, InternalCaptureControl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_DURATION_S = 3.0
YOLO_MODEL = Path(__file__).parent / ".." / "directml-benchmark" / "models" / "yolo11n.onnx"
CONFIDENCE_THRESHOLD = 0.35
APPROACH_TOLERANCE_PX = 60
CLOSE_ENOUGH_BBOX_H = 200
INPUT_SIZE = 640

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def bbox_height(self) -> float:
        return self.y2 - self.y1


@dataclass
class GameState:
    detections: list[Detection] = field(default_factory=list)
    frame_width: int = INPUT_SIZE
    frame_height: int = INPUT_SIZE


@dataclass
class Action:
    keys_press: list[str] = field(default_factory=list)
    keys_release: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FrameMetrics:
    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    detect_ms: float = 0.0
    postprocess_ms: float = 0.0
    decide_ms: float = 0.0
    act_ms: float = 0.0
    total_ms: float = 0.0
    num_detections: int = 0
    action_desc: str = ""


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    blob *= (1.0 / 255.0)
    return blob[np.newaxis]


# ---------------------------------------------------------------------------
# Detector (YOLO11n postprocess)
# ---------------------------------------------------------------------------

def postprocess_yolo(output: np.ndarray) -> list[Detection]:
    preds = output[0].T  # (8400, 84)
    boxes = preds[:, :4]
    class_scores = preds[:, 4:]
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores > CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Build detections sorted by confidence
    raw = []
    for i in range(len(scores)):
        cid = int(class_ids[i])
        raw.append(Detection(
            class_name=COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}",
            confidence=float(scores[i]),
            x1=float(x1[i]), y1=float(y1[i]),
            x2=float(x2[i]), y2=float(y2[i]),
        ))
    raw.sort(key=lambda d: d.confidence, reverse=True)

    # Greedy NMS
    kept = []
    for det in raw:
        overlap = False
        for k in kept:
            if det.class_name != k.class_name:
                continue
            ix1 = max(det.x1, k.x1)
            iy1 = max(det.y1, k.y1)
            ix2 = min(det.x2, k.x2)
            iy2 = min(det.y2, k.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (det.x2 - det.x1) * (det.y2 - det.y1)
            area_b = (k.x2 - k.x1) * (k.y2 - k.y1)
            union = area_a + area_b - inter
            if union > 0 and inter / union > 0.5:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Behavior Tree
# ---------------------------------------------------------------------------

class HasPerson(py_trees.behaviour.Behaviour):
    def __init__(self, name="See person?"):
        super().__init__(name)
        self._bb = py_trees.blackboard.Client(name=name)
        self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)

    def update(self):
        state: GameState = self._bb.game_state
        for d in state.detections:
            if d.class_name == "person" and d.confidence >= 0.35:
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class TargetCenteredAndClose(py_trees.behaviour.Behaviour):
    def __init__(self, name="Target centered & close?"):
        super().__init__(name)
        self._bb = py_trees.blackboard.Client(name=name)
        self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)

    def update(self):
        state: GameState = self._bb.game_state
        cx = state.frame_width / 2
        for d in state.detections:
            if d.class_name == "person":
                if abs(d.center_x - cx) < APPROACH_TOLERANCE_PX and d.bbox_height >= CLOSE_ENOUGH_BBOX_H:
                    return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class InteractAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="Press E"):
        super().__init__(name)
        self._bb = py_trees.blackboard.Client(name=name)
        self._bb.register_key(key="action", access=py_trees.common.Access.WRITE)

    def update(self):
        self._bb.action = Action(keys_press=["e"], description="Interact")
        return py_trees.common.Status.SUCCESS


class NavigateToTarget(py_trees.behaviour.Behaviour):
    def __init__(self, name="Navigate"):
        super().__init__(name)
        self._bb = py_trees.blackboard.Client(name=name)
        self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)
        self._bb.register_key(key="action", access=py_trees.common.Access.WRITE)

    def update(self):
        state: GameState = self._bb.game_state
        cx = state.frame_width / 2

        best = None
        for d in state.detections:
            if d.class_name == "person":
                if best is None or d.confidence > best.confidence:
                    best = d
        if best is None:
            return py_trees.common.Status.FAILURE

        keys = ["w"]
        offset = best.center_x - cx
        if offset < -APPROACH_TOLERANCE_PX:
            keys.append("left")
        elif offset > APPROACH_TOLERANCE_PX:
            keys.append("right")

        self._bb.action = Action(
            keys_press=keys,
            description=f"Chase ({best.center_x:.0f},{best.center_y:.0f}) h={best.bbox_height:.0f}",
        )
        return py_trees.common.Status.RUNNING


class WanderAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="Wander"):
        super().__init__(name)
        self._bb = py_trees.blackboard.Client(name=name)
        self._bb.register_key(key="action", access=py_trees.common.Access.WRITE)
        self._tick = 0

    def update(self):
        self._tick += 1
        keys = ["w"]
        cycle = (self._tick // 30) % 4
        if cycle == 0:
            keys.append("left")
        elif cycle == 2:
            keys.append("right")
        self._bb.action = Action(keys_press=keys, description=f"Wander cycle={cycle}")
        return py_trees.common.Status.RUNNING


def build_tree() -> py_trees.trees.BehaviourTree:
    steal = py_trees.composites.Sequence("Steal", memory=False)
    steal.add_children([HasPerson(), TargetCenteredAndClose(), InteractAction()])

    approach = py_trees.composites.Sequence("Approach", memory=False)
    approach.add_children([HasPerson("See person? (approach)"), NavigateToTarget()])

    root = py_trees.composites.Selector("Root", memory=False)
    root.add_children([steal, approach, WanderAction()])

    tree = py_trees.trees.BehaviourTree(root=root)

    bb = py_trees.blackboard.Client(name="setup")
    bb.register_key(key="game_state", access=py_trees.common.Access.WRITE)
    bb.register_key(key="action", access=py_trees.common.Access.WRITE)
    bb.game_state = GameState()
    bb.action = Action()
    return tree


# ---------------------------------------------------------------------------
# Input controller
# ---------------------------------------------------------------------------

class KeyboardController:
    def __init__(self):
        self._held: set[str] = set()
        pydirectinput.PAUSE = 0

    def execute(self, action: Action):
        to_press = set(action.keys_press)
        to_release = self._held - to_press

        for key in to_release:
            pydirectinput.keyUp(key)
        for key in to_press - self._held:
            pydirectinput.keyDown(key)

        self._held = to_press

    def release_all(self):
        for key in list(self._held):
            pydirectinput.keyUp(key)
        self._held.clear()


# ---------------------------------------------------------------------------
# Find Roblox
# ---------------------------------------------------------------------------

def find_roblox() -> str | None:
    results = []
    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and "roblox" in title.lower():
                results.append(title)
    win32gui.EnumWindows(cb, None)
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  SeMaBot2000 — End-to-End Pipeline Spike")
    print("=" * 60)

    # Find Roblox
    window_title = find_roblox()
    if not window_title:
        print("\n  ERROR: Roblox not found. Join a game first.")
        sys.exit(1)
    print(f"\n  Window: \"{window_title}\"")

    # Load YOLO
    if not YOLO_MODEL.exists():
        print(f"  ERROR: Model not found at {YOLO_MODEL}")
        sys.exit(1)

    print("  Loading YOLO11n (DirectML)...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(YOLO_MODEL), opts,
                                   providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"  Providers: {session.get_providers()}")

    # Warmup YOLO
    print("  Warming up YOLO (10 inferences)...")
    dummy = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # Build behavior tree
    tree = build_tree()
    bb = py_trees.blackboard.Client(name="pipeline")
    bb.register_key(key="game_state", access=py_trees.common.Access.WRITE)
    bb.register_key(key="action", access=py_trees.common.Access.READ)

    # Input controller
    controller = KeyboardController()

    # Shared state for pipeline
    latest_frame = [None]
    frame_lock = threading.Lock()
    frame_ready = threading.Event()
    metrics_log: list[FrameMetrics] = []
    running = threading.Event()
    running.set()

    # WGC capture thread
    capture = WindowsCapture(
        cursor_capture=None, draw_border=None,
        monitor_index=None, window_name=window_title,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        if not running.is_set():
            capture_control.stop()
            return
        img = frame.frame_buffer
        bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        with frame_lock:
            latest_frame[0] = bgr.copy()
        frame_ready.set()

    @capture.event
    def on_closed():
        frame_ready.set()

    # Start capture
    print("  Starting WGC capture...")
    cap_thread = threading.Thread(target=capture.start, daemon=True)
    cap_thread.start()

    # Wait for first frame
    frame_ready.wait(timeout=5)
    if latest_frame[0] is None:
        print("  ERROR: No frames received from WGC.")
        sys.exit(1)

    h, w = latest_frame[0].shape[:2]
    print(f"  First frame: {w}x{h}")

    # Countdown
    print(f"\n  Bot will chase the nearest person for {RUN_DURATION_S}s.")
    print("  Switch to Roblox NOW!")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1)
    print("    GO!\n")

    # === MAIN LOOP ===
    start_time = time.perf_counter()
    frame_count = 0

    try:
        while (time.perf_counter() - start_time) < RUN_DURATION_S:
            m = FrameMetrics()
            t_total = time.perf_counter()

            # Capture: grab latest frame
            t0 = time.perf_counter()
            frame_ready.clear()
            frame_ready.wait(timeout=0.1)
            with frame_lock:
                frame = latest_frame[0]
            if frame is None:
                continue
            m.capture_ms = (time.perf_counter() - t0) * 1000

            # Preprocess
            t0 = time.perf_counter()
            blob = preprocess(frame)
            m.preprocess_ms = (time.perf_counter() - t0) * 1000

            # Detect
            t0 = time.perf_counter()
            output = session.run(None, {input_name: blob})
            m.detect_ms = (time.perf_counter() - t0) * 1000

            # Postprocess
            t0 = time.perf_counter()
            detections = postprocess_yolo(output[0])
            m.postprocess_ms = (time.perf_counter() - t0) * 1000
            m.num_detections = len(detections)

            # Decide
            t0 = time.perf_counter()
            bb.game_state = GameState(detections=detections, frame_width=INPUT_SIZE, frame_height=INPUT_SIZE)
            tree.tick()
            action: Action = bb.action
            m.decide_ms = (time.perf_counter() - t0) * 1000
            m.action_desc = action.description

            # Act
            t0 = time.perf_counter()
            controller.execute(action)
            m.act_ms = (time.perf_counter() - t0) * 1000

            m.total_ms = (time.perf_counter() - t_total) * 1000
            metrics_log.append(m)
            frame_count += 1

    finally:
        # Stop everything
        running.clear()
        controller.release_all()

    elapsed = time.perf_counter() - start_time

    # === REPORT ===
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Frames processed: {frame_count}")
    print(f"  Effective FPS: {frame_count / elapsed:.1f}")

    if not metrics_log:
        print("  No metrics collected.")
        return

    def stats(values):
        avg = statistics.mean(values)
        med = statistics.median(values)
        p95 = sorted(values)[int(len(values) * 0.95)]
        return avg, med, p95

    stages = [
        ("Capture (wait)", [m.capture_ms for m in metrics_log]),
        ("Preprocess", [m.preprocess_ms for m in metrics_log]),
        ("Detect (YOLO)", [m.detect_ms for m in metrics_log]),
        ("Postprocess", [m.postprocess_ms for m in metrics_log]),
        ("Decide (BT)", [m.decide_ms for m in metrics_log]),
        ("Act (keys)", [m.act_ms for m in metrics_log]),
        ("TOTAL", [m.total_ms for m in metrics_log]),
    ]

    print(f"\n  {'Stage':<20s}  {'Avg':>8s}  {'Median':>8s}  {'P95':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, values in stages:
        avg, med, p95 = stats(values)
        print(f"  {name:<20s}  {avg:7.2f}ms  {med:7.2f}ms  {p95:7.2f}ms")

    # Action distribution
    print(f"\n  Action distribution ({frame_count} frames):")
    action_counts: dict[str, int] = {}
    for m in metrics_log:
        key = m.action_desc.split("(")[0].strip() if m.action_desc else "none"
        action_counts[key] = action_counts.get(key, 0) + 1
    for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {action:30s}: {cnt:4d} ({100*cnt/frame_count:.0f}%)")

    # Detection stats
    det_counts = [m.num_detections for m in metrics_log]
    avg_det = statistics.mean(det_counts)
    print(f"\n  Avg detections per frame: {avg_det:.1f}")
    print(f"  Frames with 0 detections: {det_counts.count(0)}")

    total_avg = statistics.mean([m.total_ms for m in metrics_log])
    print(f"\n  Pipeline: {total_avg:.1f}ms avg / {1000/total_avg:.0f} FPS")
    print()


if __name__ == "__main__":
    main()
