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

### 1.3 Double-Buffered Pipeline

The orchestrator uses a double-buffered capture pattern:

```
WGC callback thread (continuous):     Processing thread (main loop):
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ Receives frames at ~60 FPS  │       │ Grabs latest frame from     │
│ Writes to shared buffer     │──────>│ buffer (no wait)            │
│ Sets frame_new event        │       │ Processes: preprocess +     │
│                             │       │   detect + decide + act     │
└─────────────────────────────┘       └─────────────────────────────┘
```

**Validated speedup**: 1.47x (15 FPS sequential -> 22 FPS double-buffered).

The capture thread fills the buffer independently of processing speed. The main loop always processes the most recent frame, never waiting for capture. This eliminates the ~27ms capture-wait bottleneck.

**Frame-skip was evaluated and rejected**: Roblox renders every frame differently (0% skip rate in testing). Hash-based duplicate detection adds overhead with zero benefit.

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

class StateBuilder(Protocol):
    """Builds game state from detections and frame metadata."""
    def build(self, detections: list[Detection], frame: np.ndarray) -> GameState: ...

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
Implements: StateBuilder protocol
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

#### CLI (cli.py)

Entry point: `python -m semabot <command> [options]`

Uses `argparse` with subcommands. Each command maps to a specific workflow.

```
semabot run [options]          — Run the bot (main usage)
semabot capture [options]      — Capture test: save screenshots from Roblox
semabot detect <image> [opts]  — Run YOLO detection on a static image
semabot record [options]       — Record gameplay frames for training datasets
semabot auto-label [options]   — Pre-label recorded frames using current YOLO model
semabot export-model [options] — Download YOLO weights and export to ONNX
```

##### `semabot run`

The primary command. Launches the full pipeline.

```
semabot run
  --game PROFILE          Game profile name (loads config/games/<PROFILE>.toml)
                          Required.
  --config PATH           Override default bot config (default: config/default.toml)
  --dry-run               Log actions without sending keyboard input
  --save-detections       Save annotated frames to output/ (every Nth frame)
  --duration SECONDS      Run for N seconds then stop (default: unlimited)
  --log-level LEVEL       Override log level (DEBUG, INFO, WARNING, ERROR)
```

Examples:
```bash
# Play "Steal a Brainrot" for real
semabot run --game steal_a_brainrot

# Observe what the bot would do without pressing keys
semabot run --game steal_a_brainrot --dry-run

# Debug: save detection frames, verbose logging, 30 second run
semabot run --game steal_a_brainrot --dry-run --save-detections --duration 30 --log-level DEBUG
```

##### `semabot capture`

Diagnostic: verify screen capture works. Saves frames as PNGs.

```
semabot capture
  --method METHOD         Capture method: "wgc" or "mss" (default: wgc)
  --count N               Number of frames to capture (default: 5)
  --output DIR            Output directory (default: output/captures/)
```

##### `semabot detect`

Diagnostic: run YOLO detection on a saved image. Useful for testing detection quality.

```
semabot detect <image_path>
  --config PATH           Bot config for model/threshold settings
  --output PATH           Save annotated image (default: prints to console)
  --threshold FLOAT       Override confidence threshold (default: from config)
```

##### `semabot record`

Record gameplay frames for building training datasets. Captures frames at a configurable interval while the user plays normally.

```
semabot record
  --output DIR            Dataset output directory (default: datasets/<timestamp>/)
  --interval MS           Capture interval in milliseconds (default: 500)
  --duration SECONDS      Recording duration (default: 60)
  --method METHOD         Capture method: "wgc" or "mss" (default: wgc)
```

Output structure:
```
datasets/<timestamp>/
  images/
    frame_000001.png
    frame_000002.png
    ...
  metadata.json           # recording info: game, resolution, frame count, timestamps
```

##### `semabot auto-label`

Pre-label recorded frames using the current YOLO model. Generates YOLO-format annotation files (.txt per image) that can then be reviewed and corrected in an annotation tool (Label Studio, CVAT).

```
semabot auto-label
  --dataset DIR           Path to recorded dataset (images/ subdirectory)
  --config PATH           Bot config for model/threshold settings
  --class-map PATH        Optional JSON mapping COCO class names to custom class IDs
  --threshold FLOAT       Confidence threshold for auto-labels (default: 0.3)
```

Output: adds `labels/` directory alongside `images/` with YOLO-format .txt files.

##### `semabot export-model`

Download YOLO11n pretrained weights and export to ONNX format.

```
semabot export-model
  --output PATH           Output path (default: models/yolo11n.onnx)
  --input-size N          Export input size (default: 640)
```

#### BotOrchestrator

The main loop. Wires all components together and runs the pipeline.

```python
class BotOrchestrator:
    def __init__(
        self,
        frame_source: FrameSource,
        preprocessor: Preprocessor,
        detector: Detector,
        state_builder: StateBuilder,
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

### 3.5 Training Pipeline (Offline)

The training pipeline runs offline and is not part of the real-time bot loop. It produces a fine-tuned ONNX model that replaces the pretrained one.

Two labeling strategies are supported:
- **Vision LLM + Grounding DINO + Autodistill (recommended)** — Claude/GPT Vision discovers what's in each scene, Grounding DINO generates precise bounding boxes from those labels, YOLO trains on the result. Zero manual annotation.
- **YOLO auto-label + manual review (fallback)** — Pre-label with COCO YOLO locally, review/correct in Label Studio or CVAT.

```
 Local machine (GTX 960M)                Remote GPU (RTX PRO 6000, 96GB VRAM)
 ┌──────────────────────────┐            ┌──────────────────────────────────┐
 │ 1. RECORD                │            │ 2. VISION LLM DISCOVERY          │
 │ semabot record           │            │ scripts/vision_discover.py       │
 │ WGC capture at interval  │── images/ ─│ Sample frames -> Claude Vision   │
 │ while user plays         │            │ "What objects are in this game?" │
 └──────────────────────────┘            │ -> candidate class labels        │
                                         │    + text prompt ontology        │
      OR (fallback):                     ├──────────────────────────────────┤
 ┌──────────────────────────┐            │ 3. GROUNDING DINO LABELING       │
 │ 2b. LOCAL AUTO-LABEL     │            │ scripts/autodistill_label.py     │
 │ semabot auto-label       │            │ Text prompts from step 2 ->     │
 │ Pre-label with COCO YOLO │            │ precise bounding boxes per frame │
 │ Review in Label Studio   │── labels/ ─│ -> YOLO-format .txt labels       │
 └──────────────────────────┘            ├──────────────────────────────────┤
                                         │ 4. TRAIN (student model)         │
                                         │ scripts/train.py                 │
                                         │ YOLO11n fine-tune on labels      │
                                         │ epochs=100, batch=64             │
                                         ├──────────────────────────────────┤
                                         │ 5. EXPORT                        │
 ┌──────────────────────────┐            │ yolo export format=onnx          │
 │ 6. DEPLOY                │◄── .onnx ──│ Copy to local machine            │
 │ Drop .onnx into models/  │            └──────────────────────────────────┘
 │ Update config class_names│
 └──────────────────────────┘
```

#### Vision LLM Discovery (Step 2)

A frontier vision model (Claude, GPT-4V) analyzes a sample of recorded frames to discover what object classes exist in the game. This is the "semantic understanding" step — the LLM understands game context that no detection model can infer.

```python
# scripts/vision_discover.py — uses Anthropic API
# Sends a sample of frames to Claude Vision and asks it to identify
# all distinct object types, producing a text prompt ontology.

import anthropic, base64, json

client = anthropic.Anthropic()

# Send ~10-20 diverse frames
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
             "data": base64.b64encode(open(frame_path, "rb").read()).decode()}},
            # ... more frames ...
            {"type": "text", "text": """Analyze these Roblox game screenshots.
            List every distinct type of object, character, or UI element you see.
            For each, provide:
            1. A short descriptive name (e.g., "player_character")
            2. A natural language description for detection (e.g., "blocky roblox character with legs")
            Output as JSON: {"class_name": "detection_prompt", ...}"""}
        ]
    }]
)

# Output: ontology.json
# {"player": "blocky roblox character", "npc_enemy": "red enemy character", ...}
```

**Why this step matters**: Grounding DINO needs good text prompts to generate accurate boxes. A human might write "roblox character", but Claude Vision can distinguish "player character with blue hat" from "NPC shopkeeper behind counter" from "enemy with red outline" — prompts that produce far better detection quality.

#### Grounding DINO Labeling (Step 3)

[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) takes the ontology from the Vision LLM and generates precise bounding boxes for every frame.

```python
# scripts/autodistill_label.py — runs on remote GPU (RTX PRO 6000)
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import json

# Load ontology discovered by Vision LLM
with open("datasets/my_recording/ontology.json") as f:
    mapping = json.load(f)

ontology = CaptionOntology(mapping)
teacher = GroundingDINO(ontology=ontology, box_threshold=0.3)
teacher.label(
    input_folder="datasets/my_recording/images/",
    output_folder="datasets/my_recording/",
)
```

Each model contributes its strength to the pipeline:

| Model | Role | Strength | Weakness |
|-------|------|----------|----------|
| **Claude Vision** | Discover classes | Understands game context, finds subtle objects | No bounding boxes |
| **Grounding DINO** | Generate boxes | Precise spatial detection from text | Needs good prompts |
| **YOLO11n** | Production inference | 26ms real-time on GTX 960M | Needs training data |

This produces YOLO-format labels with zero manual annotation. The labels can optionally be reviewed, but for many game scenarios the quality is sufficient to train directly.

Key dependencies (remote GPU only): `autodistill`, `autodistill-grounding-dino`, `anthropic` (for Vision LLM step).

#### GameplayRecorder

Captures frames from Roblox at a configurable interval while the user plays.

```
Responsibilities:
  - Use FrameSource (WGC/MSS) to capture frames
  - Save PNGs at configurable interval (default 500ms)
  - Write metadata.json with recording info
  - Create YOLO dataset directory structure (images/)

Dependencies: FrameSource, opencv-python
```

#### AutoLabeler (local fallback)

Runs the current YOLO model on recorded frames to generate initial annotations for manual review.

```
Responsibilities:
  - Load YOLO model and run detection on each image
  - Write YOLO-format .txt labels (class_id x_center y_center w h)
  - Support class mapping (COCO names -> custom class IDs)
  - Generate data.yaml for training

Dependencies: Detector, core models
```

#### Training Scripts (scripts/)

Standalone scripts for the remote GPU machine. Not part of the semabot package.

```
scripts/
  vision_discover.py      # Claude Vision class discovery -> ontology.json
  autodistill_label.py    # Grounding DINO teacher labeling from ontology
  train.py                # YOLO fine-tuning + ONNX export
```

```python
# scripts/train.py — usage on remote machine:
# python scripts/train.py --data datasets/my_dataset/data.yaml --epochs 100 --batch 64

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=64, device=0)
model.export(format="onnx", imgsz=640, opset=17, simplify=True)
```

#### Dataset Format

```
datasets/<name>/
  images/
    train/                    # 80% of frames
      frame_000001.png
      frame_000002.png
    val/                      # 20% of frames
      frame_000050.png
  labels/
    train/
      frame_000001.txt        # YOLO format: class_id cx cy w h (normalized)
      frame_000002.txt
    val/
      frame_000050.txt
  data.yaml                   # class names, paths, nc
  metadata.json               # recording info
```

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
|   |   |-- training/                  # Offline training pipeline
|   |   |   |-- __init__.py
|   |   |   |-- recorder.py           # GameplayRecorder (capture frames at interval)
|   |   |   |-- auto_labeler.py       # AutoLabeler (pre-label with existing YOLO)
|   |   |   |-- dataset.py            # Dataset utilities (YOLO format, splits, validation)
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
|   |   |-- test_cli.py
|   |   |-- test_recorder.py
|   |   |-- test_auto_labeler.py
|   |   |-- test_dataset.py
|   |-- integration/
|   |   |-- test_detection_pipeline.py
|   |   |-- test_full_pipeline.py
|
|-- docs/
|   |-- initial-research.md
|   |-- architecture.md                # This document
|   |-- implementation-plan.md         # Task breakdown with checkboxes
|
|-- datasets/                          # Recorded + annotated training data (git-ignored)
|
|-- scripts/
|   |-- check.sh                       # Quality gate (black + ruff + pytest 90%+)
|   |-- export_model.py                # Download YOLO11n weights + export to ONNX
|   |-- vision_discover.py             # Claude Vision class discovery (API call)
|   |-- autodistill_label.py           # Grounding DINO teacher labeling (remote GPU)
|   |-- train.py                       # YOLO fine-tuning + ONNX export (remote GPU)
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
   mss_source.py   detector.py        |          null_controller.py
                   state_builder.py    +------+   key_mapper.py
                        |           conditions.py
                        |           actions.py
                        |           trees.py
         +--------------+--------------+------------------+
                        |
                        v
                   core/models.py
                   core/protocols.py
                   core/config.py
                   core/constants.py

  (offline, not in real-time pipeline)
  training/recorder.py -----> FrameSource (capture/)
  training/auto_labeler.py -> Detector (intelligence/)
  training/dataset.py ------> core/
  scripts/vision_discover.py   (standalone, anthropic API)
  scripts/autodistill_label.py (standalone, remote GPU)
  scripts/train.py             (standalone, remote GPU)
```

External dependencies are isolated at the leaf level:

Real-time pipeline:
- `windows-capture` only in `wgc_source.py`
- `mss` only in `mss_source.py`
- `onnxruntime-directml` only in `detector.py`
- `pydirectinput` only in `keyboard_controller.py`
- `py_trees` only in `intelligence/behavior/`
- `opencv-python` only in `preprocessor.py`
- `pywin32` only in `wgc_source.py` (window discovery)

Offline scripts (remote GPU only, not in semabot package):
- `anthropic` only in `scripts/vision_discover.py`
- `autodistill`, `autodistill-grounding-dino` only in `scripts/autodistill_label.py`
- `ultralytics` only in `scripts/train.py` and `scripts/export_model.py`

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
| 8: Training | Record, auto-label, train on remote GPU | "Custom Vision" |

---

## 7. Testing Strategy

### 7.1 Unit Tests

Each component is tested in isolation with mock/fake dependencies.

| Module | Test Focus |
|--------|------------|
| `models.py` | BoundingBox math, immutability, equality |
| `config.py` | TOML parsing, missing keys, game profile loading |
| `wgc_source.py` | Lifecycle, thread safety, mocked capture |
| `mss_source.py` | Mocked grab, region calculation |
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
| `cli.py` | Argument parsing for each subcommand |
| `recorder.py` | Interval timing, output structure, metadata |
| `auto_labeler.py` | Label format, class mapping, threshold filtering |
| `dataset.py` | Train/val split ratios, yaml output, path validation |

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
| Code quality | black + ruff + flake8 (cognitive complexity <=12) + pytest (90%+ coverage) | Enforced via `scripts/check.sh` at every task |
| Screen capture | WGC via `windows-capture` | DXGI blocked by Byfron; WGC works at 60 FPS |
| Object detection | YOLO11n ONNX | Best accuracy/speed for 4GB VRAM GPU; 26ms |
| GPU inference | ONNX Runtime + DirectML | Bypasses CUDA/Maxwell compatibility issues |
| Decision framework | py_trees behavior tree | 0.08ms tick; modular; industry standard for game AI |
| Input simulation | pydirectinput (SendInput) | DirectInput scan codes accepted by Roblox |
| Preprocessing | OpenCV (cv2.resize) | 9.3x faster than PIL; native numpy |
| Configuration | TOML files | Simple, readable, Python stdlib (tomllib) |
| Input method | Keyboard only | Mouse movement blocked by Roblox Byfron |
| Training labeling | Claude Vision + Grounding DINO + Autodistill | Zero manual annotation; LLM discovers classes, DINO generates boxes |

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
