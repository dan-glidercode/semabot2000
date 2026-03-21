# SeMaBot2000 — Implementation Plan

> Design details: [architecture.md](architecture.md). Benchmark data: [initial-research.md](initial-research.md).

## Quality Gate

Run before marking any task complete:

```bash
bash scripts/check.sh          # check all
bash scripts/check.sh --fix    # auto-fix formatting + lint
```

Enforces: **black** (formatting) + **ruff** (lint) + **pytest 90%+ coverage**.

Dependencies managed with **Poetry**. Common commands:
```bash
poetry install                 # install all deps (including dev group)
poetry install --with gpu      # include GPU/Windows deps
poetry run pytest              # run tests through Poetry's venv
poetry add <package>           # add a dependency
poetry add --group dev <pkg>   # add a dev dependency
```

---

## Phase 0: Project Scaffolding

- [ ] **0.1** `pyproject.toml` via Poetry — metadata, dependency groups (main/dev/gpu), tool config for black/ruff/pytest. GPU group includes: onnxruntime-directml, windows-capture, pydirectinput, py-trees, pywin32, mss
- [ ] **0.2** Directory structure — `src/semabot/{core,capture,intelligence,intelligence/behavior,action,app}/`, `tests/{unit,integration}/`, `config/games/`, `models/`
- [ ] **0.3** `.gitignore` — models/, .venv/, __pycache__/, *.onnx, captures/, *.log
- [ ] **0.4** `tests/conftest.py` + placeholder test, verify `check.sh` passes
- [ ] **0.5** `README.md` — setup instructions (`poetry install`), how to run quality gate, how to obtain YOLO model

> **Milestone 0: "Clean Slate"** — `poetry install` works, `check.sh` passes, `poetry run pytest` runs green.

---

## Phase 1: Core Layer

- [ ] **1.1** `core/models.py` — BoundingBox, Detection, GameState, Action (frozen dataclasses)
- [ ] **1.2** `tests/unit/test_models.py` — math, immutability, edge cases
- [ ] **1.3** `core/protocols.py` — FrameSource, Preprocessor, Detector, DecisionEngine, InputController (incl. release_all)
- [ ] **1.4** `core/config.py` — BotConfig, GameProfile, TOML loader
- [ ] **1.4b** `core/constants.py` — COCO_CLASSES list (80 class names), shared constants
- [ ] **1.5** `tests/unit/test_config.py` — parse TOML, missing keys, game profile
- [ ] **1.6** `config/default.toml` + `config/games/steal_a_brainrot.toml`
- [ ] **1.7** `check.sh` passes

> **Milestone 1: "Core Domain"** — Models, protocols, config with 90%+ coverage. Zero external deps.

---

## Phase 2: Capture + Action Layers

- [ ] **2.1** `capture/wgc_source.py` — WGCFrameSource (background thread, BGRA->BGR, thread-safe buffer)
- [ ] **2.2** `tests/unit/test_wgc_source.py` — lifecycle, thread safety, mocked capture
- [ ] **2.3** `capture/mss_source.py` — MSSFrameSource (fallback)
- [ ] **2.4** `tests/unit/test_mss_source.py`
- [ ] **2.5** `action/keyboard_controller.py` — KeyboardController (diff-based press/release, state tracking)
- [ ] **2.6** `tests/unit/test_keyboard_controller.py` — state transitions, release_all, mocked pydirectinput
- [ ] **2.7** `action/null_controller.py` — NullInputController for --dry-run (logs actions, sends nothing)
- [ ] **2.8** `action/key_mapper.py` — GameProfile controls to key strings
- [ ] **2.9** `tests/unit/test_key_mapper.py`
- [ ] **2.10** `check.sh` passes

> **Milestone 2: "Eyes and Hands"** — Capture saves Roblox screenshots. Keyboard makes character jump.

---

## Phase 3: Detection Pipeline

- [ ] **3.1** `intelligence/preprocessor.py` — YoloPreprocessor (cv2.resize + NCHW float32)
- [ ] **3.2** `tests/unit/test_preprocessor.py` — shape, dtype, value range, determinism
- [ ] **3.3** `intelligence/detector.py` — YoloDetector (ONNX load, inference, NMS, Detection output)
- [ ] **3.4** `tests/unit/test_detector.py` — mocked ONNX session, NMS, threshold filtering
- [ ] **3.5** `intelligence/state_builder.py` — GameStateBuilder (detections + metadata -> GameState)
- [ ] **3.6** `tests/unit/test_state_builder.py` — filtering, timestamp, immutability
- [ ] **3.7** `check.sh` passes

> **Milestone 3: "The Bot Can See"** — Saved Roblox screenshot -> YOLO -> annotated image with "person" detections.

---

## Phase 4: Behavior Tree

- [ ] **4.1** `intelligence/behavior/conditions.py` — HasDetection, TargetInCenter, TargetClose
- [ ] **4.2** `tests/unit/test_behavior_conditions.py` — true/false/edge cases per condition
- [ ] **4.3** `intelligence/behavior/actions.py` — NavigateToTarget, InteractAction, WanderAction
- [ ] **4.4** `tests/unit/test_behavior_actions.py` — correct keys per scenario
- [ ] **4.5** `intelligence/behavior/trees.py` — build_steal_a_brainrot_tree(profile)
- [ ] **4.6** `intelligence/behavior/engine.py` — BehaviorTreeEngine (blackboard, decide(state)->Action)
- [ ] **4.7** `tests/unit/test_behavior_engine.py` — 5 scenarios matching spike results
- [ ] **4.8** `check.sh` passes

> **Milestone 4: "The Bot Can Think"** — GameState in, correct Action out for all 5 scenarios.

---

## Phase 5: Orchestration

- [ ] **5.1** `app/orchestrator.py` — BotOrchestrator (main loop, FPS tracking, graceful shutdown)
- [ ] **5.2** `app/factory.py` — create_bot() wires all components from config; dry-run uses NullInputController
- [ ] **5.3** `app/cli.py` — argparse with subcommands: `run`, `capture`, `detect`, `export-model`
- [ ] **5.4** CLI `run` subcommand — `--game`, `--config`, `--dry-run`, `--save-detections`, `--duration`, `--log-level`
- [ ] **5.5** CLI `capture` subcommand — `--method`, `--count`, `--output` (diagnostic: save screenshots)
- [ ] **5.6** CLI `detect` subcommand — positional image path, `--output`, `--threshold` (diagnostic: detect on static image)
- [ ] **5.7** CLI `export-model` subcommand — `--output`, `--input-size` (download weights + export ONNX)
- [ ] **5.8** `semabot/__main__.py` — delegates to cli.py
- [ ] **5.9** `tests/unit/test_orchestrator.py` — mock components, call order, shutdown
- [ ] **5.10** `tests/unit/test_factory.py` — correct component types from config, dry-run variant
- [ ] **5.11** `tests/unit/test_cli.py` — argument parsing for each subcommand
- [ ] **5.12** `tests/integration/test_detection_pipeline.py` — preprocessor + detector on saved screenshots
- [ ] **5.13** `tests/integration/test_full_pipeline.py` — all components with synthetic frames (no Roblox)
- [ ] **5.14** `check.sh` passes

> **Milestone 5: "It's Alive"** — `semabot run --game steal_a_brainrot` launches the bot. `--dry-run` logs actions. `semabot capture` saves screenshots. `semabot detect` annotates a static image. All subcommands work.

---

## Phase 6: Robustness

- [ ] **6.1** Structured logging — per-frame metrics, startup info, errors
- [ ] **6.2** Performance metrics — rolling FPS, per-stage latency, periodic summary
- [ ] **6.3** Graceful lifecycle — signal handlers, release_all on exit, WGC cleanup, capture retry
- [ ] **6.4** `--save-detections` CLI flag — annotated frames to output/
- [ ] **6.5** Error recovery — window not found retry, model missing message, capture re-acquire
- [ ] **6.6** `check.sh` passes

> **Milestone 6: "Rock Solid"** — 10+ minute stable run. Clean Ctrl+C. Metrics logged. Detection frames saved.

---

## Phase 7: Optimization

- [ ] **7.1** Double-buffered capture — capture thread fills while processing thread works
- [ ] **7.2** Benchmark — measure FPS improvement vs sequential baseline
- [ ] **7.3** Frame-skip — reuse detections when frame unchanged
- [ ] **7.4** `check.sh` passes

> **Milestone 7: "Full Speed"** — 20+ FPS (up from 15 FPS sequential). Benchmark comparison logged.

---

## Phase 8: Custom Training Pipeline

### 8a: Data Collection (local)

- [ ] **8.1** `training/recorder.py` — GameplayRecorder (WGC capture at interval, saves PNGs + metadata.json)
- [ ] **8.2** `tests/unit/test_recorder.py` — interval timing, output structure, metadata
- [ ] **8.3** CLI `record` subcommand — `--output`, `--interval`, `--duration`, `--method`

### 8b: Labeling — Local Fallback

- [ ] **8.4** `training/auto_labeler.py` — AutoLabeler (run YOLO on images, write YOLO-format .txt labels)
- [ ] **8.5** `tests/unit/test_auto_labeler.py` — label format, class mapping, threshold filtering
- [ ] **8.6** CLI `auto-label` subcommand — `--dataset`, `--config`, `--class-map`, `--threshold`

### 8c: Labeling — Autodistill / Grounding DINO (remote GPU, recommended)

- [ ] **8.7** `scripts/autodistill_label.py` — Grounding DINO teacher labels from text prompts (zero manual annotation)
- [ ] **8.8** Ontology config — define text prompt -> class name mapping per game (JSON or in script)

### 8d: Dataset + Training

- [ ] **8.9** `training/dataset.py` — dataset utilities (train/val split, data.yaml generation, validation)
- [ ] **8.10** `tests/unit/test_dataset.py` — split ratios, yaml output, path validation
- [ ] **8.11** `scripts/train.py` — YOLO fine-tuning + ONNX export (remote GPU)
- [ ] **8.12** Documentation — training workflow in README (record -> autodistill label -> train -> deploy)
- [ ] **8.13** `check.sh` passes

> **Milestone 8: "Custom Vision"** — Record 60s of gameplay. Grounding DINO auto-labels all frames from text prompts (zero manual annotation). Train YOLO11n on remote GPU. Deploy fine-tuned ONNX model. Bot detects game-specific objects (players, NPCs, items).

---

## Summary

| Phase | Tasks | Milestone |
|-------|-------|-----------|
| 0: Scaffolding | 5 | "Clean Slate" — tooling works |
| 1: Core | 8 | "Core Domain" — models + protocols + constants |
| 2: Capture + Action | 10 | "Eyes and Hands" — capture + keyboard + dry-run |
| 3: Detection | 7 | "The Bot Can See" — YOLO pipeline |
| 4: Behavior | 8 | "The Bot Can Think" — BT decisions |
| 5: Orchestration | 14 | "It's Alive" — full bot runs |
| 6: Robustness | 6 | "Rock Solid" — stable + observable |
| 7: Optimization | 4 | "Full Speed" — 20+ FPS |
| 8: Training | 13 | "Custom Vision" — Grounding DINO labels + fine-tuned YOLO |
| **Total** | **75** | **9 milestones** |
