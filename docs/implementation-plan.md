# SeMaBot2000 — Implementation Plan

> Design details: [architecture.md](architecture.md). Benchmark data: [initial-research.md](initial-research.md).

## Quality Gate

Run before marking any task complete:

```bash
bash scripts/check.sh          # check all
bash scripts/check.sh --fix    # auto-fix formatting + lint
```

Enforces: **black** (formatting) + **ruff** (lint) + **flake8 cognitive complexity <=12** + **pytest 90%+ coverage**.

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

- [x] **0.1** `pyproject.toml` via Poetry — metadata, dependency groups (main/dev/gpu), tool config for black/ruff/pytest. Dev includes: black, ruff, pytest, pytest-cov, flake8, flake8-cognitive-complexity. GPU group includes: onnxruntime-directml, windows-capture, pydirectinput, py-trees, pywin32, mss
- [x] **0.2** Directory structure — `src/semabot/{core,capture,intelligence,intelligence/behavior,action,app,training}/`, `tests/{unit,integration}/`, `config/games/`, `models/`
- [x] **0.3** `.gitignore` — .venv/, __pycache__/, captures/, datasets/, *.log (models kept in repo)
- [x] **0.4** `tests/conftest.py` + placeholder test, verify `check.sh` passes
- [x] **0.5** `README.md` — setup instructions (`poetry install`), how to run quality gate, how to obtain YOLO model

> **Milestone 0: "Clean Slate"** — `poetry install` works, `check.sh` passes, `poetry run pytest` runs green. **DONE**

---

## Phase 1: Core Layer

- [x] **1.1** `core/models.py` — BoundingBox, Detection, GameState, Action (frozen dataclasses)
- [x] **1.2** `tests/unit/test_models.py` — math, immutability, edge cases
- [x] **1.3** `core/protocols.py` — FrameSource, Preprocessor, Detector, DecisionEngine, InputController (incl. release_all)
- [x] **1.4** `core/config.py` — BotConfig, GameProfile, TOML loader
- [x] **1.4b** `core/constants.py` — COCO_CLASSES list (80 class names), shared constants
- [x] **1.5** `tests/unit/test_config.py` — parse TOML, missing keys, game profile
- [x] **1.6** `config/default.toml` + `config/games/steal_a_brainrot.toml`
- [x] **1.7** `check.sh` passes

> **Milestone 1: "Core Domain"** — Models, protocols, config with 90%+ coverage. Zero external deps. **DONE**

---

## Phase 2: Capture + Action Layers

- [x] **2.1** `capture/wgc_source.py` — WGCFrameSource (background thread, BGRA->BGR, thread-safe buffer)
- [x] **2.2** `tests/unit/test_wgc_source.py` — lifecycle, thread safety, mocked capture
- [x] **2.3** `capture/mss_source.py` — MSSFrameSource (fallback)
- [x] **2.4** `tests/unit/test_mss_source.py`
- [x] **2.5** `action/keyboard_controller.py` — KeyboardController (diff-based press/release, state tracking)
- [x] **2.6** `tests/unit/test_keyboard_controller.py` — state transitions, release_all, mocked pydirectinput
- [x] **2.7** `action/null_controller.py` — NullInputController for --dry-run (logs actions, sends nothing)
- [x] **2.8** `action/key_mapper.py` — GameProfile controls to key strings
- [x] **2.9** `tests/unit/test_key_mapper.py`
- [x] **2.10** `check.sh` passes

> **Milestone 2: "Eyes and Hands"** — Capture saves Roblox screenshots. Keyboard makes character jump. **DONE**

---

## Phase 3: Detection Pipeline

- [x] **3.1** `intelligence/preprocessor.py` — YoloPreprocessor (cv2.resize + NCHW float32)
- [x] **3.2** `tests/unit/test_preprocessor.py` — shape, dtype, value range, determinism
- [x] **3.3** `intelligence/detector.py` — YoloDetector (ONNX load, inference, NMS, Detection output)
- [x] **3.4** `tests/unit/test_detector.py` — mocked ONNX session, NMS, threshold filtering
- [x] **3.5** `intelligence/state_builder.py` — GameStateBuilder (detections + metadata -> GameState)
- [x] **3.6** `tests/unit/test_state_builder.py` — filtering, timestamp, immutability
- [x] **3.7** `check.sh` passes

> **Milestone 3: "The Bot Can See"** — Saved Roblox screenshot -> YOLO -> annotated image with "person" detections. **DONE**

---

## Phase 4: Behavior Tree

- [x] **4.1** `intelligence/behavior/conditions.py` — HasDetection, TargetInCenter, TargetClose
- [x] **4.2** `tests/unit/test_behavior_conditions.py` — true/false/edge cases per condition
- [x] **4.3** `intelligence/behavior/actions.py` — NavigateToTarget, InteractAction, WanderAction
- [x] **4.4** `tests/unit/test_behavior_actions.py` — correct keys per scenario
- [x] **4.5** `intelligence/behavior/trees.py` — build_steal_a_brainrot_tree(profile)
- [x] **4.6** `intelligence/behavior/engine.py` — BehaviorTreeEngine (blackboard, decide(state)->Action)
- [x] **4.7** `tests/unit/test_behavior_engine.py` — 5 scenarios matching spike results
- [x] **4.8** `check.sh` passes

> **Milestone 4: "The Bot Can Think"** — GameState in, correct Action out for all 5 scenarios. **DONE**

---

## Phase 5: Orchestration

- [x] **5.1** `app/orchestrator.py` — BotOrchestrator (main loop, FPS tracking, graceful shutdown)
- [x] **5.2** `app/factory.py` — create_bot() wires all components from config; dry-run uses NullInputController
- [x] **5.3** `app/cli.py` — argparse with subcommands: `run`, `capture`, `detect`, `export-model`
- [x] **5.4** CLI `run` subcommand — `--game`, `--config`, `--dry-run`, `--save-detections`, `--duration`, `--log-level`
- [x] **5.5** CLI `capture` subcommand — `--method`, `--count`, `--output` (diagnostic: save screenshots)
- [x] **5.6** CLI `detect` subcommand — positional image path, `--output`, `--threshold` (diagnostic: detect on static image)
- [x] **5.7** CLI `export-model` subcommand — `--output`, `--input-size` (download weights + export ONNX)
- [x] **5.8** `semabot/__main__.py` — delegates to cli.py
- [x] **5.9** `tests/unit/test_orchestrator.py` — mock components, call order, shutdown
- [x] **5.10** `tests/unit/test_factory.py` — correct component types from config, dry-run variant
- [x] **5.11** `tests/unit/test_cli.py` — argument parsing for each subcommand
- [x] **5.12** `tests/integration/test_detection_pipeline.py` — preprocessor + detector on saved screenshots
- [x] **5.13** `tests/integration/test_full_pipeline.py` — all components with synthetic frames (no Roblox)
- [x] **5.14** `check.sh` passes

> **Milestone 5: "It's Alive"** — `semabot run --game steal_a_brainrot` launches the bot. `--dry-run` logs actions. `semabot capture` saves screenshots. `semabot detect` annotates a static image. All subcommands work. **DONE**

---

## Phase 6: Robustness

- [x] **6.1** Structured logging — per-frame metrics, startup info, errors
- [x] **6.2** Performance metrics — rolling FPS, per-stage latency, periodic summary
- [x] **6.3** Graceful lifecycle — signal handlers, release_all on exit, WGC cleanup, capture retry
- [x] **6.4** `--save-detections` CLI flag — annotated frames to output/
- [x] **6.5** Error recovery — window not found retry, model missing message, capture re-acquire
- [x] **6.6** `check.sh` passes

> **Milestone 6: "Rock Solid"** — 10+ minute stable run. Clean Ctrl+C. Metrics logged. Detection frames saved. **DONE**

---

## Phase 7: Optimization

- [x] **7.1** Spike: benchmark double-buffer vs sequential vs frame-skip
- [x] **7.2** Implement double-buffered capture in orchestrator — grab latest frame without waiting
- [x] ~~**7.3** Frame-skip~~ — **DROPPED**: 0% skip rate in Roblox (game renders every frame differently)
- [x] **7.4** `check.sh` passes

> **Milestone 7: "Full Speed"** — 22 FPS (up from 15 FPS sequential). Double-buffer validated at 1.47x speedup in spike. **DONE**

Spike results (2026-03-22, `spikes/double-buffer/double_buffer_spike.py`):
- Sequential: 68.2ms / 15 FPS (baseline)
- Double-buffered: 46.3ms / 22 FPS (1.47x)
- Frame-skip: 0 frames skipped — useless for Roblox

---

## Phase 8: Custom Training Pipeline

### 8a: Data Collection (local)

- [x] **8.1** `training/recorder.py` — GameplayRecorder (WGC capture at interval, saves PNGs + metadata.json)
- [x] **8.2** `tests/unit/test_recorder.py` — interval timing, output structure, metadata
- [x] **8.3** CLI `record` subcommand — `--output`, `--interval`, `--duration`, `--method`

### 8b: Labeling — Local Fallback

- [x] **8.4** `training/auto_labeler.py` — AutoLabeler (run YOLO on images, write YOLO-format .txt labels)
- [x] **8.5** `tests/unit/test_auto_labeler.py` — label format, class mapping, threshold filtering
- [x] **8.6** CLI `auto-label` subcommand — `--dataset`, `--config`, `--class-map`, `--threshold`

### 8c: Labeling — Vision LLM + Grounding DINO (remote GPU, recommended)

- [ ] **8.7** `scripts/vision_discover.py` — send sample frames to Claude Vision, output ontology.json (class -> prompt mapping)
- [ ] **8.8** `scripts/autodistill_label.py` — Grounding DINO reads ontology.json, generates bounding boxes for all frames
- [ ] **8.8b** Ontology review step — validate/edit ontology.json before running Grounding DINO

### 8d: Dataset + Training

- [x] **8.9** `training/dataset.py` — dataset utilities (train/val split, data.yaml generation, validation)
- [x] **8.10** `tests/unit/test_dataset.py` — split ratios, yaml output, path validation
- [ ] **8.11** `scripts/train.py` — YOLO fine-tuning + ONNX export (remote GPU)
- [x] **8.12** Documentation — training workflow in README (record -> autodistill label -> train -> deploy)
- [ ] **8.13** `check.sh` passes

> **Milestone 8: "Custom Vision"** — Record 60s of gameplay. Claude Vision discovers object classes. Grounding DINO auto-labels all frames (zero manual annotation). Train YOLO11n on remote GPU. Deploy fine-tuned model. Bot detects game-specific objects.

---

## Summary

| Phase | Done | Total | Milestone |
|-------|------|-------|-----------|
| 0: Scaffolding | 5/5 | 5 | "Clean Slate" — DONE |
| 1: Core | 8/8 | 8 | "Core Domain" — DONE |
| 2: Capture + Action | 10/10 | 10 | "Eyes and Hands" — DONE |
| 3: Detection | 7/7 | 7 | "The Bot Can See" — DONE |
| 4: Behavior | 8/8 | 8 | "The Bot Can Think" — DONE |
| 5: Orchestration | 14/14 | 14 | "It's Alive" — DONE |
| 6: Robustness | 6/6 | 6 | "Rock Solid" — DONE |
| 7: Optimization | 4/4 | 4 | "Full Speed" — DONE (22 FPS, 1.47x) |
| 8: Training | 9/14 | 14 | "Custom Vision" — 5 tasks need API key / remote GPU |
| **Total** | **71/76** | **76** | **8/9 milestones done** |
