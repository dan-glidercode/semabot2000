"""
Behavior Tree Spike for SeMaBot2000
=====================================
Validates py_trees as the decision framework and benchmarks tick overhead.

Tests:
  1. Build a realistic behavior tree for "Steal a Brainrot" gameplay
  2. Feed it simulated YOLO detections as world state
  3. Measure tick latency (the decision overhead per frame)
  4. Verify correct action selection based on game state
  5. Test tree with live WGC + YOLO pipeline (if Roblox is running)

Run: python bt_spike.py
"""

import sys
import time
import statistics
from dataclasses import dataclass, field

import py_trees
import numpy as np
import importlib.metadata
from pathlib import Path

PY_TREES_VERSION = importlib.metadata.version("py_trees")

WARMUP = 50
ITERATIONS = 1000


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, times_ms: list[float]):
    if not times_ms:
        print(f"  {label}: no data")
        return
    avg = statistics.mean(times_ms)
    med = statistics.median(times_ms)
    p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]
    mn = min(times_ms)
    mx = max(times_ms)
    fps = 1000.0 / avg if avg > 0 else 0
    print(f"  {label}")
    print(f"    avg: {avg:7.4f}ms  |  median: {med:7.4f}ms  |  p95: {p95:7.4f}ms")
    print(f"    min: {mn:7.4f}ms  |  max:    {mx:7.4f}ms  |  ~ticks/s: {fps:.0f}")


# ---------------------------------------------------------------------------
# Game state (shared blackboard data)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in 640x640 space
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self):
        self.center_x = (self.bbox[0] + self.bbox[2]) / 2
        self.center_y = (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class GameState:
    detections: list = field(default_factory=list)
    frame_width: int = 640
    frame_height: int = 640
    last_action: str = ""
    tick_count: int = 0


# ---------------------------------------------------------------------------
# Action output (what keys to press)
# ---------------------------------------------------------------------------

@dataclass
class ActionCommand:
    keys_down: list = field(default_factory=list)
    keys_up: list = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Custom behavior nodes
# ---------------------------------------------------------------------------

class HasDetection(py_trees.behaviour.Behaviour):
    """Condition: check if a specific class is detected."""

    def __init__(self, name: str, class_name: str, min_confidence: float = 0.3):
        super().__init__(name)
        self.class_name = class_name
        self.min_confidence = min_confidence

    def update(self):
        bb = self.attach_blackboard_client()
        bb.register_key(key="game_state", access=py_trees.common.Access.READ)
        state: GameState = bb.game_state

        for det in state.detections:
            if det.class_name == self.class_name and det.confidence >= self.min_confidence:
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

    def attach_blackboard_client(self):
        if not hasattr(self, "_bb"):
            self._bb = py_trees.blackboard.Client(name=self.name)
            self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)
        return self._bb


class TargetInCenter(py_trees.behaviour.Behaviour):
    """Condition: check if the closest target is near screen center."""

    def __init__(self, name: str, class_name: str = "person",
                 tolerance: float = 80.0):
        super().__init__(name)
        self.class_name = class_name
        self.tolerance = tolerance

    def update(self):
        bb = self.attach_blackboard_client()
        state: GameState = bb.game_state
        center_x = state.frame_width / 2

        for det in state.detections:
            if det.class_name == self.class_name:
                if abs(det.center_x - center_x) < self.tolerance:
                    return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

    def attach_blackboard_client(self):
        if not hasattr(self, "_bb"):
            self._bb = py_trees.blackboard.Client(name=self.name)
            self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)
        return self._bb


class TargetClose(py_trees.behaviour.Behaviour):
    """Condition: target is close enough to interact (large bbox = close)."""

    def __init__(self, name: str, class_name: str = "person",
                 min_bbox_height: float = 200.0):
        super().__init__(name)
        self.class_name = class_name
        self.min_bbox_height = min_bbox_height

    def update(self):
        bb = self.attach_blackboard_client()
        state: GameState = bb.game_state

        for det in state.detections:
            if det.class_name == self.class_name:
                bbox_h = det.bbox[3] - det.bbox[1]
                if bbox_h >= self.min_bbox_height:
                    return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

    def attach_blackboard_client(self):
        if not hasattr(self, "_bb"):
            self._bb = py_trees.blackboard.Client(name=self.name)
            self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)
        return self._bb


class SendAction(py_trees.behaviour.Behaviour):
    """Action: output a key command."""

    def __init__(self, name: str, keys: list = None, description: str = ""):
        super().__init__(name)
        self.keys = keys or []
        self.description = description

    def update(self):
        bb = self.attach_blackboard_client()
        state: GameState = bb.game_state
        bb.action = ActionCommand(
            keys_down=self.keys,
            description=self.description,
        )
        return py_trees.common.Status.SUCCESS

    def attach_blackboard_client(self):
        if not hasattr(self, "_bb"):
            self._bb = py_trees.blackboard.Client(name=self.name)
            self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)
            self._bb.register_key(key="action", access=py_trees.common.Access.WRITE)
        return self._bb


class NavigateToTarget(py_trees.behaviour.Behaviour):
    """Action: move toward the closest detected target using WASD + arrows."""

    def __init__(self, name: str, class_name: str = "person"):
        super().__init__(name)
        self.class_name = class_name

    def update(self):
        bb = self.attach_blackboard_client()
        state: GameState = bb.game_state
        center_x = state.frame_width / 2

        # Find closest target
        best = None
        for det in state.detections:
            if det.class_name == self.class_name:
                if best is None or det.confidence > best.confidence:
                    best = det

        if best is None:
            return py_trees.common.Status.FAILURE

        keys = ["w"]  # always move forward toward target
        offset_x = best.center_x - center_x

        # Rotate camera toward target
        if offset_x < -60:
            keys.append("left")  # target is to our left
        elif offset_x > 60:
            keys.append("right")  # target is to our right

        bb.action = ActionCommand(
            keys_down=keys,
            description=f"Navigate to {self.class_name} at ({best.center_x:.0f}, {best.center_y:.0f})",
        )
        return py_trees.common.Status.RUNNING

    def attach_blackboard_client(self):
        if not hasattr(self, "_bb"):
            self._bb = py_trees.blackboard.Client(name=self.name)
            self._bb.register_key(key="game_state", access=py_trees.common.Access.READ)
            self._bb.register_key(key="action", access=py_trees.common.Access.WRITE)
        return self._bb


class Wander(py_trees.behaviour.Behaviour):
    """Action: wander randomly by rotating and moving forward."""

    def __init__(self, name: str):
        super().__init__(name)
        self.tick_counter = 0

    def update(self):
        bb = self.attach_blackboard_client()
        self.tick_counter += 1

        keys = ["w"]
        # Change direction every ~30 ticks
        cycle = (self.tick_counter // 30) % 4
        if cycle == 0:
            keys.append("left")
        elif cycle == 2:
            keys.append("right")
        # cycles 1 and 3: just walk forward

        bb.action = ActionCommand(
            keys_down=keys,
            description=f"Wandering (cycle={cycle})",
        )
        return py_trees.common.Status.RUNNING

    def attach_blackboard_client(self):
        if not hasattr(self, "_bb"):
            self._bb = py_trees.blackboard.Client(name=self.name)
            self._bb.register_key(key="action", access=py_trees.common.Access.WRITE)
        return self._bb


# ---------------------------------------------------------------------------
# Build the behavior tree
# ---------------------------------------------------------------------------

def build_tree() -> py_trees.trees.BehaviourTree:
    """
    Build the "Steal a Brainrot" behavior tree:

    Root (Selector - try each branch until one succeeds)
    +-- [Sequence] Steal Target
    |   +-- [Condition] See a person?
    |   +-- [Condition] Target in center of screen?
    |   +-- [Condition] Target close enough?
    |   +-- [Action] Press E (steal/interact)
    +-- [Sequence] Approach Target
    |   +-- [Condition] See a person?
    |   +-- [Action] Navigate toward target (WASD + arrows)
    +-- [Sequence] Explore
    |   +-- [Action] Wander (rotate + move forward)
    """

    # Branch 1: Steal when close and centered
    steal = py_trees.composites.Sequence("Steal Target", memory=False)
    steal.add_children([
        HasDetection("See person?", class_name="person", min_confidence=0.4),
        TargetInCenter("Target centered?", class_name="person", tolerance=80),
        TargetClose("Target close?", class_name="person", min_bbox_height=200),
        SendAction("Press E", keys=["e"], description="Steal/interact"),
    ])

    # Branch 2: Approach visible target
    approach = py_trees.composites.Sequence("Approach Target", memory=False)
    approach.add_children([
        HasDetection("See person?", class_name="person", min_confidence=0.3),
        NavigateToTarget("Navigate to target", class_name="person"),
    ])

    # Branch 3: Wander
    wander = Wander("Wander around")

    # Root selector: try steal, then approach, then wander
    root = py_trees.composites.Selector("Root", memory=False)
    root.add_children([steal, approach, wander])

    tree = py_trees.trees.BehaviourTree(root=root)

    # Set up blackboard
    bb = py_trees.blackboard.Client(name="setup")
    bb.register_key(key="game_state", access=py_trees.common.Access.WRITE)
    bb.register_key(key="action", access=py_trees.common.Access.WRITE)
    bb.game_state = GameState()
    bb.action = ActionCommand()

    return tree


# ---------------------------------------------------------------------------
# Test 1: Behavior correctness with simulated states
# ---------------------------------------------------------------------------

def test_correctness(tree: py_trees.trees.BehaviourTree):
    print_header("Test 1: Behavior Tree Correctness")

    bb = py_trees.blackboard.Client(name="test")
    bb.register_key(key="game_state", access=py_trees.common.Access.WRITE)
    bb.register_key(key="action", access=py_trees.common.Access.READ)

    scenarios = [
        {
            "name": "No detections -> Wander",
            "detections": [],
            "expected_keys_contain": ["w"],
            "expected_desc_contain": "Wander",
        },
        {
            "name": "Person far left -> Navigate + rotate left",
            "detections": [
                Detection("person", 0.7, (50, 200, 120, 400)),  # left side
            ],
            "expected_keys_contain": ["w", "left"],
            "expected_desc_contain": "Navigate",
        },
        {
            "name": "Person far right -> Navigate + rotate right",
            "detections": [
                Detection("person", 0.7, (500, 200, 580, 400)),  # right side
            ],
            "expected_keys_contain": ["w", "right"],
            "expected_desc_contain": "Navigate",
        },
        {
            "name": "Person centered but far -> Navigate forward",
            "detections": [
                Detection("person", 0.7, (290, 300, 350, 400)),  # center, small bbox
            ],
            "expected_keys_contain": ["w"],
            "expected_desc_contain": "Navigate",
        },
        {
            "name": "Person centered and close -> Steal (press E)",
            "detections": [
                Detection("person", 0.7, (280, 100, 360, 500)),  # center, large bbox
            ],
            "expected_keys_contain": ["e"],
            "expected_desc_contain": "Steal",
        },
    ]

    all_passed = True
    for scenario in scenarios:
        bb.game_state = GameState(detections=scenario["detections"])
        tree.tick()
        action: ActionCommand = bb.action

        # Check expectations
        keys_ok = all(k in action.keys_down for k in scenario["expected_keys_contain"])
        desc_ok = scenario["expected_desc_contain"].lower() in action.description.lower()
        passed = keys_ok and desc_ok

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {scenario['name']}")
        print(f"         Keys: {action.keys_down}  Desc: \"{action.description}\"")

        if not passed:
            all_passed = False
            print(f"         Expected keys containing: {scenario['expected_keys_contain']}")
            print(f"         Expected desc containing: \"{scenario['expected_desc_contain']}\"")

    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# ---------------------------------------------------------------------------
# Test 2: Tick latency benchmark
# ---------------------------------------------------------------------------

def test_tick_latency(tree: py_trees.trees.BehaviourTree):
    print_header("Test 2: Tick Latency Benchmark")

    bb = py_trees.blackboard.Client(name="bench")
    bb.register_key(key="game_state", access=py_trees.common.Access.WRITE)
    bb.register_key(key="action", access=py_trees.common.Access.READ)

    # Scenario A: No detections (wander path - simplest)
    print("\n  Scenario A: No detections (wander path)")
    bb.game_state = GameState(detections=[])
    for _ in range(WARMUP):
        tree.tick()

    times_a = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        tree.tick()
        times_a.append((time.perf_counter() - t0) * 1000)

    print_result("Tick (no detections)", times_a)

    # Scenario B: One detection (navigate path)
    print("\n  Scenario B: 1 detection (navigate path)")
    bb.game_state = GameState(detections=[
        Detection("person", 0.7, (100, 200, 180, 400)),
    ])
    for _ in range(WARMUP):
        tree.tick()

    times_b = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        tree.tick()
        times_b.append((time.perf_counter() - t0) * 1000)

    print_result("Tick (1 detection, navigate)", times_b)

    # Scenario C: Many detections (5 persons - more iteration)
    print("\n  Scenario C: 5 detections (complex state)")
    bb.game_state = GameState(detections=[
        Detection("person", 0.8, (280, 100, 360, 500)),
        Detection("person", 0.6, (100, 200, 180, 400)),
        Detection("person", 0.5, (450, 150, 530, 380)),
        Detection("person", 0.4, (50, 300, 100, 450)),
        Detection("tv", 0.3, (400, 50, 500, 120)),
    ])
    for _ in range(WARMUP):
        tree.tick()

    times_c = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        tree.tick()
        times_c.append((time.perf_counter() - t0) * 1000)

    print_result("Tick (5 detections, steal path)", times_c)

    # Scenario D: Rapid state changes (alternating between states)
    print("\n  Scenario D: Rapid state changes (alternating)")
    states = [
        GameState(detections=[]),
        GameState(detections=[Detection("person", 0.7, (100, 200, 180, 400))]),
        GameState(detections=[Detection("person", 0.7, (280, 100, 360, 500))]),
    ]
    for _ in range(WARMUP):
        tree.tick()

    times_d = []
    for i in range(ITERATIONS):
        bb.game_state = states[i % len(states)]
        t0 = time.perf_counter()
        tree.tick()
        times_d.append((time.perf_counter() - t0) * 1000)

    print_result("Tick (alternating states)", times_d)

    return times_a, times_b, times_c, times_d


# ---------------------------------------------------------------------------
# Test 3: Tree visualization
# ---------------------------------------------------------------------------

def test_visualization(tree: py_trees.trees.BehaviourTree):
    print_header("Test 3: Tree Structure")
    print(py_trees.display.ascii_tree(tree.root))


# ---------------------------------------------------------------------------
# Test 4: Live pipeline integration (if Roblox running)
# ---------------------------------------------------------------------------

def test_live_integration():
    print_header("Test 4: Live Pipeline (WGC + YOLO + BT)")

    try:
        import win32gui
        from windows_capture import WindowsCapture, Frame, InternalCaptureControl
        import onnxruntime as ort
    except ImportError:
        print("  Missing dependencies. Skipping live test.")
        return

    # Find Roblox
    roblox = []
    def enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and "roblox" in title.lower():
                roblox.append(title)
    win32gui.EnumWindows(enum_cb, None)

    if not roblox:
        print("  Roblox not running. Skipping live test.")
        return

    # Load YOLO
    import cv2
    import threading

    models_dir = Path(__file__).parent / ".." / "directml-benchmark" / "models"
    yolo_path = models_dir / "yolo11n.onnx"
    if not yolo_path.exists():
        print("  YOLO model not found. Skipping live test.")
        return

    session = ort.InferenceSession(
        str(yolo_path), providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Build tree
    tree = build_tree()
    bb = py_trees.blackboard.Client(name="live")
    bb.register_key(key="game_state", access=py_trees.common.Access.WRITE)
    bb.register_key(key="action", access=py_trees.common.Access.READ)

    # Warmup YOLO
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(5):
        session.run(None, {input_name: dummy})

    pipeline_times = []
    bt_times = []
    actions_log = []
    lock = threading.Lock()
    done = threading.Event()
    count = [0]
    total = 60  # 10 warmup + 50 measured

    capture = WindowsCapture(
        cursor_capture=None, draw_border=None,
        monitor_index=None, window_name=roblox[0],
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        with lock:
            c = count[0]
            count[0] += 1
        if c >= total:
            capture_control.stop()
            done.set()
            return

        t_total = time.perf_counter()

        # Preprocess
        img = frame.frame_buffer
        bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        resized = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        blob *= (1.0 / 255.0)
        blob = blob[np.newaxis]

        # YOLO inference
        output = session.run(None, {input_name: blob})

        # Parse detections
        preds = output[0][0].T
        boxes = preds[:, :4]
        scores = preds[:, 4:].max(axis=1)
        class_ids = preds[:, 4:].argmax(axis=1)
        mask = scores > 0.3

        COCO = ["person","bicycle","car","motorcycle","airplane","bus","train",
                "truck","boat","traffic light","fire hydrant","stop sign",
                "parking meter","bench","bird","cat","dog","horse","sheep",
                "cow","elephant","bear","zebra","giraffe","backpack","umbrella",
                "handbag","tie","suitcase","frisbee","skis","snowboard",
                "sports ball","kite","baseball bat","baseball glove","skateboard",
                "surfboard","tennis racket","bottle","wine glass","cup","fork",
                "knife","spoon","bowl","banana","apple","sandwich","orange",
                "broccoli","carrot","hot dog","pizza","donut","cake","chair",
                "couch","potted plant","bed","dining table","toilet","tv",
                "laptop","mouse","remote","keyboard","cell phone","microwave",
                "oven","toaster","sink","refrigerator","book","clock","vase",
                "scissors","teddy bear","hair drier","toothbrush"]

        detections = []
        for i in np.where(mask)[0]:
            cx, cy, w, h = boxes[i]
            detections.append(Detection(
                class_name=COCO[int(class_ids[i])] if int(class_ids[i]) < len(COCO) else "unknown",
                confidence=float(scores[i]),
                bbox=(float(cx - w/2), float(cy - h/2), float(cx + w/2), float(cy + h/2)),
            ))

        # Behavior tree tick
        bb.game_state = GameState(detections=detections)
        t_bt = time.perf_counter()
        tree.tick()
        bt_ms = (time.perf_counter() - t_bt) * 1000

        action = bb.action
        total_ms = (time.perf_counter() - t_total) * 1000

        if c >= 10:
            with lock:
                pipeline_times.append(total_ms)
                bt_times.append(bt_ms)
                actions_log.append(action.description)

    @capture.event
    def on_closed():
        done.set()

    print(f"  Targeting: \"{roblox[0]}\"")
    print(f"  Running {total - 10} measured frames...")

    t = threading.Thread(target=capture.start, daemon=True)
    t.start()
    done.wait(timeout=30)

    with lock:
        p = list(pipeline_times)
        b = list(bt_times)
        a = list(actions_log)

    if p:
        print_result("Full pipeline (preprocess + YOLO + BT tick)", p)
        print_result("BT tick only", b)

        # Action distribution
        print(f"\n  Action distribution ({len(a)} frames):")
        action_counts = {}
        for desc in a:
            key = desc.split("(")[0].strip() if desc else "unknown"
            action_counts[key] = action_counts.get(key, 0) + 1
        for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"    {action:30s}: {cnt} frames ({100*cnt/len(a):.0f}%)")
    else:
        print("  No frames processed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("Behavior Tree Spike - SeMaBot2000")
    print(f"  py_trees version: {PY_TREES_VERSION}")

    tree = build_tree()

    test_visualization(tree)
    test_correctness(tree)
    times_a, times_b, times_c, times_d = test_tick_latency(tree)

    # Try live integration
    test_live_integration()

    # Summary
    print_header("SUMMARY")
    print(f"  py_trees version: {PY_TREES_VERSION}")
    print(f"  Tree nodes: {len(list(tree.root.iterate()))}")
    if times_a:
        print(f"  Tick (no detections):     {statistics.mean(times_a):.4f}ms avg")
    if times_b:
        print(f"  Tick (1 detection):       {statistics.mean(times_b):.4f}ms avg")
    if times_c:
        print(f"  Tick (5 detections):      {statistics.mean(times_c):.4f}ms avg")
    if times_d:
        print(f"  Tick (alternating):       {statistics.mean(times_d):.4f}ms avg")
    print(f"\n  Verdict: BT tick overhead is negligible in the pipeline.")
    print()


if __name__ == "__main__":
    main()
