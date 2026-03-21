# SeMaBot2000 — Architecture Document

## 1. Overview

SeMaBot2000 is an AI-powered Roblox bot that plays games autonomously using computer vision, GPU-accelerated object detection, behavior tree decision-making, and keyboard input simulation.

The system observes the game through screen capture, understands the scene via YOLO11n object detection, decides what to do through a behavior tree, and acts through simulated keyboard input. All components have been individually validated through spike benchmarks.

### 1.1 Design Principles

- **Separation of Concerns** — Each pipeline stage is an independent module with a well-defined interface. Capture knows nothing about detection; detection knows nothing about decision-making.
- **Dependency Inversion** — Core logic depends on abstractions (protocols/interfaces), not concrete implementations. Swapping WGC for MSS, or YOLO for NanoDet, requires no changes to the pipeline orchestrator.
- **Single Responsibility** — Each class/module does one thing. A detector detects. A controller sends keys. A behavior tree decides.
- **Open/Closed** — New game strategies are added by extending the behavior tree with new nodes, not by modifying existing ones. New detectors implement the `Detector` protocol.
- **Interface Segregation** — Consumers depend only on the slice of functionality they need. The behavior tree sees `GameState`, not raw frames or ONNX sessions.
- **Composition over Inheritance** — The pipeline is assembled from composable components, not deep class hierarchies.

### 1.2 Validated Performance Budget

| Stage | Component | Measured Latency | Hardware |
|-------|-----------|-----------------|----------|
| Capture | WGC (windows-capture) | 17ms avg / 60 FPS | CPU |
| Preprocess | cv2.resize + NCHW | 8.7ms avg | CPU |
| Detect | YOLO11n ONNX + DirectML | 26ms avg (under load) | GPU (GTX 960M) |
| Postprocess | NMS + decode | 3.2ms avg | CPU |
| Decide | py_trees behavior tree | 0.26ms avg | CPU |
| Act | pydirectinput SendInput | 0.8ms avg | CPU |
| **Total processing** | **(stages 2-6)** | **37.6ms / 27 FPS** | |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
+------------------------------------------------------------------+
|                        SeMaBot2000                                |
|                                                                   |
|  +------------------+    +------------------------------------+   |
|  | Capture Layer    |    |        Processing Pipeline         |   |
|  |                  |    |                                    |   |
|  | WGC Window       |--->| Preprocessor -> Detector ->        |   |
|  | Capture          |    | PostProcessor -> GameStateBuilder  |   |
|  | (event-driven,   |    +------------------------------------+   |
|  |  background      |                    |                        |
|  |  thread)         |                    v                        |
|  +------------------+    +------------------------------------+   |
|                          |        Decision Layer              |   |
|                          |                                    |   |
|                          | BehaviorTree (py_trees)            |   |
|                          | reads GameState, writes Action     |   |
|                          +------------------------------------+   |
|                                          |                        |
|                                          v                        |
|                          +------------------------------------+   |
|                          |        Action Layer                |   |
|                          |                                    |   |
|                          | InputController (pydirectinput)    |   |
|                          | translates Action -> key presses   |   |
|                          +------------------------------------+   |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                    Cross-Cutting Concerns                   |  |
|  | Configuration | Logging | Metrics | Game Profile Registry  |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### 2.2 Layered Architecture

The system is organized into four horizontal layers, each with clear upward dependencies only.

```
    +---------------------------------------------------------+
    |                   Application Layer                      |
    |  BotOrchestrator, CLI entry point, lifecycle management  |
    +---------------------------------------------------------+
                |               |                |
    +-----------v-+   +---------v--------+   +---v-----------+
    | Capture     |   | Intelligence     |   | Action        |
    | Layer       |   | Layer            |   | Layer         |
    |             |   |                  |   |               |
    | FrameSource |   | Preprocessor     |   | InputCtrl     |
    |             |   | Detector         |   | KeyMapper     |
    |             |   | GameStateBuilder |   |               |
    |             |   | BehaviorTree     |   |               |
    +-------------+   +------------------+   +---------------+
                |               |                |
    +-----------v---------------v----------------v-----------+
    |                     Core Layer                          |
    |  Domain models: Frame, Detection, GameState, Action     |
    |  Protocols/interfaces, Configuration, Shared types      |
    +---------------------------------------------------------+
```

**Dependency Rule**: Each layer may only depend on layers below it. The Core layer has zero external dependencies.

### 2.3 Core Domain Models

```python
# core/models.py — Pure data, no behavior, no external deps

@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]: ...
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...
    @property
    def area(self) -> float: ...

@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox

@dataclass(frozen=True)
class GameState:
    detections: tuple[Detection, ...]
    frame_width: int
    frame_height: int
    timestamp: float

@dataclass(frozen=True)
class Action:
    keys_press: tuple[str, ...]    # keys to hold this frame
    keys_release: tuple[str, ...]  # keys to release this frame
    description: str = ""
```

### 2.4 Protocol Definitions (Interfaces)

```python
# core/protocols.py — Abstract interfaces, no implementations

class FrameSource(Protocol):
    """Provides raw frames from the game."""
    def start(self) -> None: ...
    def get_latest_frame(self) -> np.ndarray | None: ...
    def stop(self) -> None: ...

class Preprocessor(Protocol):
    """Transforms a raw frame into model input."""
    def process(self, frame: np.ndarray) -> np.ndarray: ...

class Detector(Protocol):
    """Runs object detection on a preprocessed frame."""
    def detect(self, blob: np.ndarray) -> list[Detection]: ...

class DecisionEngine(Protocol):
    """Decides what action to take given the current game state."""
    def decide(self, state: GameState) -> Action: ...

class InputController(Protocol):
    """Sends actions to the game."""
    def execute(self, action: Action) -> None: ...
    def release_all(self) -> None: ...
```

Any module that implements one of these protocols can be plugged into the pipeline. This is how we swap implementations without changing orchestration logic.

---

## 3. Component Design

### 3.1 Capture Layer

#### WGCFrameSource

Wraps `windows-capture` library. Runs WGC in a background thread, stores the latest frame in a thread-safe buffer.

```
Responsibilities:
  - Find and target the Roblox window by title
  - Receive frames via WGC callback
  - Convert BGRA -> BGR
  - Expose latest frame via get_latest_frame()

Dependencies: windows-capture, numpy, win32gui
Implements: FrameSource protocol
```

#### MSSFrameSource (fallback)

GDI BitBlt capture via python-mss. Synchronous grab, slower but guaranteed to work.

```
Implements: FrameSource protocol
```

### 3.2 Intelligence Layer

#### YoloPreprocessor

Resizes and normalizes frames for YOLO11n input.

```
Responsibilities:
  - cv2.resize to 640x640
  - Transpose HWC -> CHW, normalize to [0, 1] float32
  - Return contiguous (1, 3, 640, 640) array

Dependencies: opencv-python, numpy
Implements: Preprocessor protocol
```

#### YoloDetector

Runs YOLO11n ONNX model via DirectML, parses raw output into Detection objects.

```
Responsibilities:
  - Load ONNX model, create inference session
  - Run inference
  - Decode output tensor: extract boxes, scores, class IDs
  - Apply NMS
  - Return list[Detection]

Dependencies: onnxruntime-directml
Implements: Detector protocol
```

#### GameStateBuilder

Transforms raw detections + frame metadata into a GameState object consumed by the behavior tree.

```
Responsibilities:
  - Filter detections by confidence threshold
  - Attach frame dimensions and timestamp
  - Build immutable GameState

Dependencies: core models only
```

#### BehaviorTreeEngine

Wraps py_trees to provide the DecisionEngine interface. Manages the blackboard and tree lifecycle.

```
Responsibilities:
  - Build and configure the behavior tree
  - Write GameState to blackboard before tick
  - Read Action from blackboard after tick
  - Expose decide(state) -> Action

Dependencies: py_trees, core models
Implements: DecisionEngine protocol
```

### 3.3 Action Layer

#### KeyboardController

Translates Action objects into pydirectinput key presses/releases. Tracks currently-held keys to avoid redundant press/release calls.

```
Responsibilities:
  - Track key state (pressed/released)
  - Press new keys, release old keys
  - Handle key mapping (game profile -> scan codes)

Dependencies: pydirectinput
Implements: InputController protocol
```

#### NullInputController

No-op implementation for `--dry-run` mode. Logs actions without sending any input.

```
Responsibilities:
  - Implement InputController protocol
  - Log action descriptions (keys that would be pressed)
  - Send nothing to the OS

Dependencies: none (logging only)
Implements: InputController protocol
```

### 3.4 Application Layer

#### BotOrchestrator

The main loop. Wires all components together and runs the pipeline.

```python
class BotOrchestrator:
    def __init__(
        self,
        frame_source: FrameSource,
        preprocessor: Preprocessor,
        detector: Detector,
        state_builder: GameStateBuilder,
        decision_engine: DecisionEngine,
        input_controller: InputController,
        config: BotConfig,
    ): ...

    def run(self) -> None:
        """Main loop: capture -> preprocess -> detect -> build state -> decide -> act."""
        self.frame_source.start()
        try:
            while self._running:
                frame = self.frame_source.get_latest_frame()
                if frame is None:
                    continue

                blob = self.preprocessor.process(frame)
                detections = self.detector.detect(blob)
                state = self.state_builder.build(detections, frame)
                action = self.decision_engine.decide(state)
                self.input_controller.execute(action)
        finally:
            self.frame_source.stop()
            self.input_controller.release_all()
```

The orchestrator owns no business logic. It coordinates. All intelligence lives in the components it composes.

---

## 4. Project Structure

```
SeMaBot2000/
|-- src/
|   |-- semabot/
|   |   |-- __init__.py
|   |   |
|   |   |-- core/                      # Core layer (zero external deps)
|   |   |   |-- __init__.py
|   |   |   |-- models.py             # BoundingBox, Detection, GameState, Action
|   |   |   |-- protocols.py          # FrameSource, Detector, DecisionEngine, etc.
|   |   |   |-- config.py             # BotConfig, GameProfile dataclasses
|   |   |   |-- constants.py          # COCO_CLASSES list, shared constants
|   |   |
|   |   |-- capture/                   # Capture layer
|   |   |   |-- __init__.py
|   |   |   |-- wgc_source.py         # WGCFrameSource (windows-capture)
|   |   |   |-- mss_source.py         # MSSFrameSource (fallback)
|   |   |
|   |   |-- intelligence/             # Intelligence layer
|   |   |   |-- __init__.py
|   |   |   |-- preprocessor.py       # YoloPreprocessor
|   |   |   |-- detector.py           # YoloDetector (ONNX + DirectML)
|   |   |   |-- state_builder.py      # GameStateBuilder
|   |   |   |-- behavior/             # Behavior tree
|   |   |   |   |-- __init__.py
|   |   |   |   |-- engine.py         # BehaviorTreeEngine
|   |   |   |   |-- conditions.py     # HasDetection, TargetInCenter, etc.
|   |   |   |   |-- actions.py        # NavigateToTarget, Interact, Wander
|   |   |   |   |-- trees.py          # Tree builders per game profile
|   |   |
|   |   |-- action/                    # Action layer
|   |   |   |-- __init__.py
|   |   |   |-- keyboard_controller.py # KeyboardController (pydirectinput)
|   |   |   |-- null_controller.py    # NullInputController (dry-run, logging only)
|   |   |   |-- key_mapper.py         # Game-specific key bindings
|   |   |
|   |   |-- app/                       # Application layer
|   |   |   |-- __init__.py
|   |   |   |-- orchestrator.py        # BotOrchestrator
|   |   |   |-- cli.py                 # CLI entry point
|   |   |   |-- factory.py            # Component assembly / DI
|   |
|-- models/                            # ONNX model files (git-ignored)
|   |-- yolo11n.onnx
|
|-- config/                            # Configuration files
|   |-- default.toml                   # Default bot config
|   |-- games/
|   |   |-- steal_a_brainrot.toml     # Game-specific profile
|
|-- tests/
|   |-- conftest.py                    # Shared fixtures (GameState factories, mock sessions)
|   |-- unit/
|   |   |-- test_models.py
|   |   |-- test_config.py
|   |   |-- test_preprocessor.py
|   |   |-- test_detector.py
|   |   |-- test_state_builder.py
|   |   |-- test_behavior_conditions.py
|   |   |-- test_behavior_actions.py
|   |   |-- test_behavior_engine.py
|   |   |-- test_wgc_source.py
|   |   |-- test_mss_source.py
|   |   |-- test_keyboard_controller.py
|   |   |-- test_key_mapper.py
|   |   |-- test_orchestrator.py
|   |   |-- test_factory.py
|   |-- integration/
|   |   |-- test_detection_pipeline.py
|   |   |-- test_full_pipeline.py
|
|-- docs/
|   |-- initial-research.md
|   |-- architecture.md                # This document
|
|-- scripts/
|   |-- check.sh                       # Quality gate (black + ruff + pytest 90%+)
|   |-- export_model.py                # Download YOLO11n weights + export to ONNX
|
|-- spikes/                            # Validated spike benchmarks
|
|-- pyproject.toml                     # Poetry project config + tool settings
|-- poetry.lock                        # Locked dependency versions
|-- README.md
```

### 4.1 Dependency Graph

```
  cli.py
    |
    v
  factory.py -----> orchestrator.py
                        |
         +--------------+--------------+------------------+
         |              |              |                   |
         v              v              v                   v
   wgc_source.py   preprocessor.py  engine.py    keyboard_controller.py
   mss_source.py   detector.py      conditions.py null_controller.py
                   state_builder.py  actions.py    key_mapper.py
         |              |              |                   |
         +--------------+--------------+------------------+
                        |
                        v
                   core/models.py
                   core/protocols.py
                   core/config.py
                   core/constants.py
```

External dependencies are isolated at the leaf level:
- `windows-capture` only in `wgc_source.py`
- `mss` only in `mss_source.py`
- `onnxruntime-directml` only in `detector.py`
- `pydirectinput` only in `keyboard_controller.py`
- `py_trees` only in `intelligence/behavior/`
- `opencv-python` only in `preprocessor.py`
- `pywin32` only in `wgc_source.py` (window discovery)

---

## 5. Configuration

### 5.1 Bot Configuration (default.toml)

```toml
[bot]
target_fps = 25
log_level = "INFO"

[capture]
method = "wgc"               # "wgc" or "mss"
window_title = "Roblox"

[detection]
model_path = "models/yolo11n.onnx"
provider = "DmlExecutionProvider"
confidence_threshold = 0.35
input_size = 640
nms_iou_threshold = 0.5

[decision]
tree_tick_rate_hz = 25        # match target FPS

[action]
method = "pydirectinput"
key_hold_duration_ms = 50     # min time to hold a key per frame
```

### 5.2 Game Profile (steal_a_brainrot.toml)

```toml
[game]
name = "Steal a Brainrot"
window_title = "Roblox"

[controls]
move_forward = "w"
move_backward = "s"
move_left = "a"
move_right = "d"
camera_left = "left"
camera_right = "right"
jump = "space"
interact = "e"

[detection]
target_classes = ["person"]
target_min_confidence = 0.4

[behavior]
approach_tolerance_px = 60    # pixel offset to consider "centered"
close_enough_bbox_height = 200  # bbox height threshold for "close"
wander_cycle_ticks = 30       # ticks per wander direction change
```

---

## 6. Implementation Phases

See [implementation-plan.md](implementation-plan.md) for the detailed task breakdown with checkboxes and milestones.

| Phase | Goal | Milestone |
|-------|------|-----------|
| 0: Scaffolding | Poetry, tooling, directory structure | "Clean Slate" |
| 1: Core | Domain models, protocols, config | "Core Domain" |
| 2: Capture + Action | WGC/MSS capture, keyboard controller | "Eyes and Hands" |
| 3: Detection | YOLO preprocessor, detector, state builder | "The Bot Can See" |
| 4: Behavior | BT conditions, actions, engine | "The Bot Can Think" |
| 5: Orchestration | BotOrchestrator, factory, CLI | "It's Alive" |
| 6: Robustness | Logging, metrics, error recovery | "Rock Solid" |
| 7: Optimization | Double-buffered capture, frame-skip | "Full Speed" |

---

## 7. Testing Strategy

### 7.1 Unit Tests

Each component is tested in isolation with mock/fake dependencies.

| Module | Test Focus |
|--------|------------|
| `models.py` | BoundingBox math, immutability, equality |
| `config.py` | TOML parsing, missing keys, game profile loading |
| `preprocessor.py` | Output shape, dtype, value range, determinism |
| `detector.py` | Mock ONNX session, verify Detection output structure |
| `state_builder.py` | Filtering, timestamp, immutability |
| `conditions.py` | Each BT condition with crafted GameState |
| `actions.py` | Each BT action produces correct Action output |
| `engine.py` | Full tree tick, correct Action for each scenario |
| `keyboard_controller.py` | Key state tracking, press/release sequencing |
| `key_mapper.py` | Control name to key string mapping, validation |
| `orchestrator.py` | Mock components, call order, shutdown, None frame skip |
| `factory.py` | Correct component types from config, dry-run variant |

### 7.2 Integration Tests

Test multi-component flows with real dependencies (but not Roblox).

| Test | Components | Notes |
|------|------------|-------|
| Detection pipeline | Preprocessor -> Detector -> GameStateBuilder | Uses saved screenshots |
| Full pipeline | All components with synthetic frames | No Roblox needed, uses NullInputController |

### 7.3 Manual Validation

| Test | Method |
|------|--------|
| End-to-end gameplay | Run bot with Roblox, observe character behavior |
| Detection accuracy | Save annotated frames, visual inspection |
| Performance | Log FPS and per-stage latency over 60s run |

---

## 8. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dependency management | Poetry | Lockfile for reproducible builds; dependency groups (main/dev/gpu); `pyproject.toml`-native |
| Screen capture | WGC via `windows-capture` | DXGI blocked by Byfron; WGC works at 60 FPS |
| Object detection | YOLO11n ONNX | Best accuracy/speed for 4GB VRAM GPU; 26ms |
| GPU inference | ONNX Runtime + DirectML | Bypasses CUDA/Maxwell compatibility issues |
| Decision framework | py_trees behavior tree | 0.08ms tick; modular; industry standard for game AI |
| Input simulation | pydirectinput (SendInput) | DirectInput scan codes accepted by Roblox |
| Preprocessing | OpenCV (cv2.resize) | 9.3x faster than PIL; native numpy |
| Configuration | TOML files | Simple, readable, Python stdlib (tomllib) |
| Input method | Keyboard only | Mouse movement blocked by Roblox Byfron |

---

## 9. Constraints and Risks

| Risk | Mitigation |
|------|------------|
| Byfron anti-cheat update blocks WGC | MSS fallback ready; monitor Roblox updates |
| Byfron blocks SendInput keyboard | No known mitigation; fundamental blocker |
| YOLO pretrained model insufficient | Custom training on labeled Roblox screenshots |
| GTX 960M VRAM exhaustion | Monitor VRAM; YOLO11n uses <200MB; plenty of headroom |
| py_trees API changes | Pin version in pyproject.toml |
| Game-specific behavior too rigid | Game profile system allows per-game config without code changes |
