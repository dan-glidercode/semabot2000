# SeMaBot2000

AI-powered Roblox bot that plays games autonomously using computer vision, GPU-accelerated object detection (YOLO11n via DirectML), behavior tree decision-making (py-trees), and keyboard input simulation (pydirectinput).

The system observes the game through screen capture, understands the scene via YOLO11n object detection, decides what to do through a behavior tree, and acts through simulated keyboard input.

## Requirements

- Python >= 3.12
- Poetry (package manager)
- Windows 10/11 (required for DirectML, WGC capture, and pydirectinput)
- A DirectX 12-capable GPU (for ONNX Runtime DirectML inference)

## Setup

1. **Install Poetry** if you don't already have it:

   ```bash
   pip install poetry
   ```

2. **Install project dependencies**:

   ```bash
   # Core + dev dependencies (sufficient for tests and linting)
   poetry install

   # Include GPU runtime dependencies (needed to actually run the bot)
   poetry install --with gpu
   ```

3. **Obtain the YOLO model** (see section below).

## Obtaining the YOLO Model

The bot requires a YOLO11n ONNX model placed at `models/yolo11n.onnx`. This file is not checked into version control due to its size.

**Option A: Export from Ultralytics (recommended)**

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True
```

Then copy the resulting `yolo11n.onnx` into the `models/` directory.

**Option B: Use a fine-tuned model**

If you have trained a custom YOLO model for your target game (see the training pipeline in `docs/architecture.md`), place the exported `.onnx` file in `models/` and update the model path in your config file.

## Running the Quality Gate

The quality gate checks formatting, linting, and tests:

```bash
# Format check
poetry run black --check src/ tests/

# Lint
poetry run ruff check src/ tests/

# Run tests with coverage
poetry run pytest --cov=semabot --cov-report=term-missing
```

All three must pass before merging code.

## Project Structure

```
src/semabot/          Main package (src layout)
  core/               Domain models, protocols, config (zero external deps)
  capture/            Screen capture (WGC, MSS fallback)
  intelligence/       Preprocessing, YOLO detection, game state, behavior tree
  action/             Keyboard input simulation
  app/                CLI entry point, orchestrator, component factory
  training/           Offline recording and auto-labeling tools
tests/                Test suite (unit + integration)
config/               TOML configuration files
config/games/         Per-game profiles
models/               ONNX model files (git-ignored, see above)
docs/                 Architecture and planning documents
scripts/              Standalone scripts (training, model export)
```

## Usage

```bash
# Run the bot on a game
poetry run python -m semabot run --game steal_a_brainrot

# Dry-run mode (observe without sending input)
poetry run python -m semabot run --game steal_a_brainrot --dry-run

# Capture test frames
poetry run python -m semabot capture --count 5

# Detect objects in a static image
poetry run python -m semabot detect screenshot.png
```

See `docs/architecture.md` for the full architecture, component design, and training pipeline documentation.
