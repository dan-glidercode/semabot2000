# Initial Research: AI-Powered Roblox Bot (SeMaBot2000)

## Goal

Build a Python-based AI bot that controls a Roblox character using local GPU-accelerated AI tools.

---

## Architecture Overview

There are two primary architectural approaches, which can also be combined:

### Approach A: Computer Vision + Input Simulation (External)

Python runs externally, captures the game screen, processes it with AI/CV models on the GPU, and sends keyboard/mouse inputs back to the Roblox client.

```
[Roblox Client] --screen capture--> [Python CV Pipeline (GPU)] --keyboard/mouse--> [Roblox Client]
```

### Approach B: In-Game Luau Script + External Python Server (Hybrid)

A Luau script inside Roblox communicates with a Python server via HTTP. The Python server runs AI logic and sends back commands.

```
[Roblox Luau Script] --HTTP POST/GET--> [Python Server (AI on GPU)] --JSON response--> [Roblox Luau Script]
```

### Approach C: Roblox Open Cloud API (Remote Control)

Use Roblox Open Cloud APIs from Python to interact with DataStores and MessagingService, allowing indirect control of in-game behavior.

```
[Python + rblx-open-cloud] --Open Cloud API--> [Roblox DataStore/MessagingService] --read by--> [Luau Script in-game]
```

---

## Key Components & Libraries

### 1. Screen Capture

| Library | API | Performance | Roblox Compatible |
|---------|-----|-------------|-------------------|
| **windows-capture** | Windows Graphics Capture (WGC) | **~17ms avg / ~60 FPS** | **YES — VALIDATED (recommended)** |
| **python-mss** | GDI BitBlt | ~32ms avg / ~31 FPS | **YES — VALIDATED (fallback)** |
| **DXcam** | DXGI Desktop Duplication | 240Hz+ theoretical | **NO — BLOCKED by Byfron** |
| **BetterCam** | DXGI Desktop Duplication | 240Hz+ theoretical | **NO — same API, likely blocked** |

- **CRITICAL FINDING**: DXGI Desktop Duplication is blocked by Roblox's Byfron/Hyperion anti-cheat (`COMError: Catastrophic failure`). Rules out DXcam and BetterCam.
- **windows-capture (WGC)** is the recommended capture method. It targets the Roblox window directly (no desktop/chrome), delivers frames at ~60 FPS, and is not blocked by Byfron. Event-driven API with `frame_buffer` providing numpy arrays (BGRA).
- **python-mss** (GDI BitBlt) works as a fallback but is 2x slower and captures the whole screen region.
- Install: `pip install windows-capture` (Rust-based, uses WGC API)
- GitHub: https://github.com/NiiightmareXD/windows-capture

### 2. Computer Vision & AI Models

| Tool | Use Case |
|------|----------|
| **OpenCV** | Template matching, image processing, basic object detection |
| **YOLOv8/v11 (Ultralytics)** | Real-time object detection (characters, items, UI elements) |
| **PyTorch** | Custom model training, reinforcement learning |
| **TensorRT / ONNX Runtime** | GPU-optimized inference for production speed |
| **EfficientNet** | Lightweight image classification for game state recognition |

- YOLO is particularly well-suited for detecting in-game objects without memory injection.
- PyTorch models can be trained on captured gameplay data.

### 3. Input Simulation

| Library | API | Keyboard | Mouse Move | Roblox Compatible |
|---------|-----|----------|------------|-------------------|
| **pydirectinput** | SendInput (DirectInput scan codes) | **YES** | **NO** | **YES — VALIDATED (recommended)** |
| **pynput** | SendInput (VK codes) | **YES** | **NO** | **YES — VALIDATED** |
| **ctypes SendInput** | Raw SendInput (scan codes) | **YES** | **NO** | **YES — VALIDATED** |
| **PyAutoGUI** | keybd_event / mouse_event (deprecated) | **NO** | **NO** | **NO — not accepted by Roblox** |
| **pyrobloxbot** | Keyboard-based movement | Not tested | N/A | Likely yes (keyboard only) |

- **CRITICAL FINDING**: Roblox accepts keyboard input from SendInput-based methods (pynput, pydirectinput, ctypes) but **rejects all simulated mouse movement**. Roblox likely uses Raw Input or DirectInput for mouse look, which cannot be faked by standard SendInput mouse events. The bot must be designed around **keyboard-only control**.
- **pydirectinput** is recommended — uses DirectInput scan codes via SendInput, designed for game input, and validated working with Roblox.
- **PyAutoGUI does not work** — its deprecated keybd_event/mouse_event API is not accepted by Roblox for any input type.
- **Camera/look control**: The target game ("Steal a Brainrot") supports keyboard camera rotation via Left/Right arrow keys. Full control scheme: WASD (move), Arrow keys (camera), Space (jump), E (action/purchase). This means the bot has full control without needing mouse input.
- Install: `pip install pydirectinput`

### 4. Roblox API Communication

| Library | Description |
|---------|-------------|
| **ro.py** | Async Python wrapper for Roblox web API |
| **rblx-open-cloud** | Python wrapper for Roblox Open Cloud (DataStores, MessagingService, assets) |
| **HttpService (Luau)** | In-game HTTP requests to external servers (500 req/min limit) |

- **rblx-open-cloud**: `pip install rblx-open-cloud~=2.0` - Allows Python to read/write DataStores and publish messages to in-game servers.
- GitHub: https://github.com/treeben77/rblx-open-cloud
- **HttpService**: Roblox server-side Luau can make HTTP GET/POST to an external Python Flask/FastAPI server. Rate limited to 500 requests/minute.

### 5. AI Decision Making

| Approach | Description | Status |
|----------|-------------|--------|
| **Behavior Trees (py_trees)** | Hierarchical, modular decision framework | **VALIDATED — 0.08ms tick, 10 nodes** |
| **Rule-based** | If-then logic within BT leaf nodes | Used inside BT conditions |
| **LLM Integration** | Claude API for complex/strategic decisions (async) | Planned (non-real-time) |
| **Reinforcement Learning** | Train agents to learn optimal play | Future (Phase 4) |

- **py_trees 2.4.0** is the validated decision framework. Pure Python, no heavy dependencies, ~0.08ms per tick (isolated), ~0.26ms under live pipeline load. 10-node tree handles the full "Steal a Brainrot" gameplay loop.
- Tree structure: Selector root -> Steal (conditions + action) -> Approach (navigate toward target) -> Wander (fallback)
- Blackboard pattern for sharing YOLO detections and action commands between nodes.
- Install: `pip install py_trees`
- Docs: https://py-trees.readthedocs.io/

---

## Chosen Architecture: Approach A — Computer Vision + Input Simulation

Real-time gameplay requires the external CV pipeline. The bot sees the screen, understands the game state via GPU-accelerated models, makes decisions, and acts through input simulation.

```
  +----------------+       +-------------------------------------------+
  |                |       |         Python Main Process               |
  | Roblox Client  | screen|                                           |
  | (any game)     |------>|  WGC capture ──> YOLO/CV (GPU) ──> Game  |
  |                |       |  (windows-capture) (DirectML)     State   |
  |                |       |                                 |         |
  |                | keys/ |                          AI Decision      |
  |                |<------| pynput/pyrobloxbot <──── Engine           |
  |                | mouse |                                           |
  +----------------+       +-------------------------------------------+
                                          |
                                   Claude API (optional)
                                   for complex reasoning
```

> **Note**: DXcam (DXGI) is blocked by Roblox's Byfron anti-cheat. Windows Graphics Capture API (WGC) via `windows-capture` is the validated capture method (~60 FPS, window-targeted).

### Core Loop (target: <50ms per cycle)

Estimated latencies (pre-benchmark):

```
1. CAPTURE  ─ DXcam grabs frame from Roblox window        (~4ms)
2. DETECT   ─ YOLO11n via ONNX+DirectML finds objects     (~15-30ms)
3. CLASSIFY ─ Game state from detections + OCR/templates   (~2-5ms)
4. DECIDE   ─ AI engine picks next action                  (~1-5ms)
5. ACT      ─ Send keyboard/mouse input to Roblox          (~1ms)
```

**Fully measured latencies** (end-to-end live pipeline, 2026-03-21):

```
1. CAPTURE      ─ WGC (windows-capture), event-driven       (~17ms avg inter-frame, 60 FPS)
2. PREPROCESS   ─ cv2.resize + NCHW float32                  (~8.7ms avg)
3. DETECT       ─ YOLO11n DirectML inference                  (~26ms avg under load)
4. POSTPROCESS  ─ NMS + decode detections                     (~3.2ms avg)
5. DECIDE       ─ py_trees behavior tree tick                  (~0.26ms avg)
6. ACT          ─ pydirectinput SendInput key tap              (~0.8ms avg)
```

**End-to-end processing (stages 2-5): 37.6ms avg / 27 FPS.** All stages now measured.

```
 WGC callback thread:                   Processing thread:
 ┌──────────────────────┐               ┌──────────────────────────────┐
 │ WGC frame arrives    │──latest───>   │ cv2 preprocess       ~8.7ms  │
 │ ~17ms avg (~60 FPS)  │  frame        │ YOLO11n DirectML    ~26.0ms  │
 │ (event-driven)       │               │ Post-process (NMS)   ~3.2ms  │
 └──────────────────────┘               │ Decision engine      ~1-3ms  │
                                        │ Input action         ~0.8ms  │
                                        └──────────────────────────────┘
 Processing total: ~38-40ms
 Effective rate: max(17, 40) = ~40ms / ~25 FPS
```

**YOLO11n detects Roblox characters as COCO "person" class** (0.60-0.75 confidence). Across 50 live frames, "person" appeared 168 times (multiple characters per frame) and "tv" 16 times (UI elements/signs). Custom training would improve detection of game-specific objects but the pretrained model provides a usable baseline.

MSS (GDI BitBlt) remains available as a fallback at ~31 FPS for the smaller windowed resolution.

### End-to-End Proof of Concept (2026-03-22)

Script: `spikes/end-to-end/e2e_test.py` — Full pipeline running live against Roblox for 3 seconds.

**Pipeline: WGC capture -> cv2 preprocess -> YOLO11n DirectML -> NMS -> py_trees BT -> pydirectinput**

| Stage | Avg | Median | P95 |
|-------|-----|--------|-----|
| Capture (wait for frame) | 26.7ms | 26.1ms | 30.3ms |
| Preprocess | 5.9ms | 5.2ms | 6.7ms |
| Detect (YOLO) | 28.7ms | 26.9ms | 31.3ms |
| Postprocess | 3.5ms | 3.4ms | 4.3ms |
| Decide (BT) | 0.22ms | 0.20ms | 0.31ms |
| Act (keys) | 0.30ms | 0.01ms | 0.81ms |
| **TOTAL** | **65.3ms** | **63.0ms** | **94.4ms** |

**Result: 46 frames in 3.01s = 15 FPS (sequential pipeline).**

Processing-only time (excluding capture wait): **38.7ms avg** — matches isolated benchmarks exactly.

With double-buffered capture (Phase 4 optimization): projected **~25 FPS**.

#### Bot behavior during the run:

| Action | Frames | % |
|--------|--------|---|
| Chase (navigate toward person) | 37 | 80% |
| Interact (press E) | 8 | 17% |
| Wander (no detections) | 1 | 2% |

The bot successfully detected persons, chased them, and pressed E when close enough. **Proof of concept validated** — the full pipeline works end-to-end.

#### Known limitations observed:
- **Chase tracking is coarse** — the bot tends to dash forward without turning quickly enough when the target moves laterally. The behavior tree's camera rotation logic (left/right arrow keys) needs tuning: shorter decision intervals, proportional turning speed, or predictive tracking.
- **Sequential pipeline caps FPS** — the main loop blocks waiting for each new frame. Double-buffering capture and processing would nearly double throughput.
- **Single detection class** — pretrained COCO only sees "person". Game-specific objects (items, NPCs, obstacles) need custom training.

### Phase 1: Foundation
1. Set up Python environment (ONNX Runtime + DirectML, PyTorch 2.5.x as fallback)
2. Implement screen capture with DXcam targeting Roblox window
3. Basic input simulation with pynput or pyrobloxbot
4. Build the main loop skeleton (capture → process → act)

### Phase 2: Vision
5. Integrate OpenCV for template matching and basic game state detection
6. Export YOLOv8n to ONNX, run inference via DirectML
7. Train a custom YOLO model on Roblox game screenshots (if needed)
8. Add OCR for reading in-game text (health, scores, chat)

### Phase 3: Intelligence
9. Implement rule-based decision engine for basic gameplay
10. Add behavior trees for structured action sequences
11. Integrate Claude API for complex reasoning (non-real-time decisions)

### Phase 4: Refinement
12. Optimize latency (profiling, batch processing, model quantization)
13. Add game-specific modules (navigation, combat, resource gathering)
14. Explore reinforcement learning for specific repetitive tasks

---

## Local GPU: NVIDIA GeForce GTX 960M

### Hardware Specs

| Spec | Value |
|------|-------|
| **GPU** | NVIDIA GeForce GTX 960M |
| **Architecture** | Maxwell (GM107) |
| **VRAM** | 4 GB GDDR5 |
| **CUDA Compute Capability** | 5.0 (sm_50) |
| **CUDA Cores** | 640 |
| **Driver** | 511.23 |
| **CUDA Version** | 11.6 |

### Compatibility Constraints

This is an older Maxwell-generation GPU. Modern deep learning tooling has been progressively dropping support:

- **PyTorch**: Pre-built wheels for PyTorch **2.5.x with CUDA 12.1 or 12.4** are the last versions to include Maxwell (sm_50) support. PyTorch 2.8+ with CUDA 12.8/12.9 drops Maxwell entirely. Building from source is possible but complex.
- **CUDA Toolkit**: The installed driver supports CUDA 11.6. Newer CUDA toolkits (12.8+) deprecate Maxwell.
- **TensorRT**: Latest versions require compute capability 6.1+, ruling out direct use on this GPU.

### Viable Inference Strategies

Given the 4 GB VRAM and compute capability 5.0, here are the realistic options:

#### Option 1: ONNX Runtime + DirectML (Recommended)

DirectML is a hardware-accelerated DirectX 12 library that works with **any DirectX 12-capable GPU**, including Maxwell. This sidesteps the CUDA compatibility issues entirely.

- Install: `pip install onnxruntime-directml`
- Supports running ONNX-exported models (YOLO, classification, etc.)
- No CUDA version dependency — uses DirectX 12 instead
- Slightly slower than native CUDA but much easier to set up
- Works with the GTX 960M since it supports DirectX 12

#### Option 2: PyTorch 2.5.x + CUDA 12.1

Pin to an older but still capable PyTorch version:

```
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
```

- Supports sm_50 (Maxwell)
- Gives access to full PyTorch ecosystem (Ultralytics YOLO, etc.)
- Downside: locked to an older PyTorch, missing newer features

#### Option 3: ONNX Runtime + CUDA (Older Build)

Use ONNX Runtime with CUDA execution provider matched to CUDA 11.6:

- Install: `pip install onnxruntime-gpu` (ensure version compatible with CUDA 11.x)
- Good inference speed, supports older CUDA
- Export models to ONNX format first, then run inference

#### Option 4: CPU Fallback for Heavy Models

For models too large or incompatible with the GPU:
- Run on CPU with ONNX Runtime (no GPU needed)
- Slower but no compatibility issues
- Viable for non-real-time tasks (LLM reasoning, planning)

### What Can Run on 4 GB VRAM

| Model / Task | VRAM Usage | Feasibility | Measured Performance |
|--------------|-----------|-------------|---------------------|
| **YOLO11n (nano)** | ~50-200 MB | Excellent | **21ms / 47 FPS (DirectML), 90ms / 11 FPS (CPU)** |
| **YOLOv8s (small)** | ~200-400 MB | Good | Not yet benchmarked — expect ~2x slower than nano |
| **MobileNetV3-Large** | ~30-80 MB | Excellent | **10ms / 98 FPS (DirectML), 9ms / 110 FPS (CPU)** |
| **EfficientNet-B0** | ~50-100 MB | Excellent | Not yet benchmarked — similar class to MobileNet |
| **TinyLlama 1.1B (Q4)** | ~1-2 GB | Possible | Simple text reasoning, ~60 tokens/s (est.) |
| **SmolLM2 1.7B (Q4)** | ~1.5-2.5 GB | Tight | Better reasoning, may compete for VRAM |
| **Phi-3.5 Mini 3.8B (Q3)** | ~3-3.5 GB | Borderline | Leaves little room for anything else |
| **OpenCV template matching** | Minimal | Excellent | CPU-based, no VRAM needed |

### Recommended GPU Strategy for SeMaBot2000

```
Primary path:   ONNX Runtime + DirectML  (VALIDATED — 4.2x speedup over CPU for YOLO11n)
Fallback path:  PyTorch 2.5.x + CUDA 12.1 (not needed — DirectML performance is sufficient)
LLM reasoning:  CPU-based or API calls (Claude API for complex decisions)
```

**Practical split (updated with benchmark data):**
- **Real-time detection** (YOLO11n via ONNX + DirectML) → GPU, **21ms / 47 FPS** — confirmed viable
- **Game state classification** (MobileNetV3 via ONNX) → **run on CPU** (~9ms, faster than GPU due to DirectML overhead on small models)
- **Complex reasoning** (LLM) → offload to Claude API or run TinyLlama on CPU when GPU is busy
- **Screen capture** (DXcam) → minimal GPU overhead, runs alongside inference
- **Preprocessing** → use OpenCV `cv2.resize` instead of PIL (PIL is the current bottleneck)

---

## Important Considerations

- **Anti-Cheat**: Roblox has anti-cheat systems (Byfron/Hyperion). External screen-reading and input simulation is generally safer than memory injection, but still carries risk. Use responsibly.
- **Terms of Service**: Automation may violate Roblox ToS depending on usage context. This project is for educational/research purposes.
- **Rate Limits**: HttpService is limited to 500 external requests/minute. Open Cloud has its own rate limits.
- **GPU Requirements**: Local GPU is a GTX 960M (4 GB, Maxwell, sm_50). Use ONNX Runtime + DirectML as primary inference path; pin PyTorch to 2.5.x if CUDA is needed. See GPU section above.
- **Latency**: The capture → process → act loop should target <50ms for responsive bot behavior. Pure YOLO11n inference via DirectML achieves 21ms, but the full pipeline (capture + preprocess + infer) currently measures ~96ms. Optimization of preprocessing is the main path to improvement.

---

## Benchmark Results — GTX 960M Validation

Benchmark run: 2026-03-21. Script: `spikes/directml-benchmark/benchmark.py`

### Environment

| Component | Version |
|-----------|---------|
| ONNX Runtime | 1.24.4 (DirectML) |
| Python | 3.14.3 |
| NumPy | 2.4.3 |
| GPU | GTX 960M 4GB (Maxwell, sm_50) |
| CPU | Intel Core i7-4720HQ 2.60GHz |

### Isolated Model Inference (pure inference, no preprocessing)

| Model | Provider | Avg | Median | P95 | Min | Max | FPS |
|-------|----------|-----|--------|-----|-----|-----|-----|
| **YOLO11n @ 640x640** | DirectML | **21.3ms** | 21.2ms | 21.9ms | 20.9ms | 22.1ms | **47** |
| YOLO11n @ 640x640 | CPU | 89.9ms | 89.4ms | 96.7ms | 83.3ms | 102.6ms | 11 |
| **MobileNetV3-Large @ 224x224** | DirectML | 10.2ms | 10.1ms | 11.5ms | 9.4ms | 12.5ms | 98 |
| MobileNetV3-Large @ 224x224 | CPU | **9.1ms** | 9.1ms | 10.4ms | 7.4ms | 11.0ms | **110** |

### Full Pipeline (capture + preprocess + infer)

| Stage | Avg | Median | P95 | FPS |
|-------|-----|--------|-----|-----|
| Screen capture (synthetic fallback) | 13.2ms | 11.4ms | 21.4ms | 76 |
| YOLO11n detect + preprocess | 57.8ms | 58.6ms | 65.2ms | 17 |
| MobileNetV3 classify + preprocess | 25.2ms | 25.2ms | 30.7ms | 40 |
| **Total pipeline** | **96.2ms** | 97.6ms | 110.5ms | **~10** |

### Screen Capture — Roblox Validation

Benchmark run: 2026-03-21. Script: `spikes/dxcam-roblox-capture/capture_fallback_test.py`

Tested with live Roblox game window (1744x1059). Screenshots verified to contain real game content.

| Method | Avg | Median | P95 | Min | Max | Status |
|--------|-----|--------|-----|-----|-----|--------|
| **windows-capture (WGC)** | **16.8ms** | 2.0ms | 34.6ms | 0.8ms | 92.9ms | **Works (recommended)** |
| python-mss (GDI BitBlt) | 32.3ms | 31.4ms | 33.6ms | 26.0ms | 63.4ms | Works (fallback) |
| DXcam (DXGI Duplication) | N/A | N/A | N/A | N/A | N/A | **BLOCKED by Byfron** |
| Win32 PrintWindow | N/A | N/A | N/A | N/A | N/A | API unavailable |

WGC frame preprocess (in callback, BGRA->BGR + cv2.resize + NCHW float32): **10.1ms avg, 12.2ms p95**

Note: WGC captures window-only (802x631) vs MSS full region (1744x1059). Smaller frames = faster preprocessing.

### Key Findings

1. **DirectML delivers 4.2x speedup** for YOLO11n over CPU (21ms vs 90ms). GTX 960M is viable for detection via DirectML.
2. **MobileNetV3 is faster on CPU** (9ms) than DirectML (10ms). **Run classification on CPU.**
3. **OpenCV preprocessing is 9.3x faster than PIL** (3ms vs 28ms). Use `cv2.resize` + `ascontiguousarray`.
4. **DXGI Desktop Duplication is blocked by Roblox Byfron anti-cheat.** DXcam and BetterCam cannot be used.
5. **Windows Graphics Capture API (WGC) works and is fast** — ~60 FPS, ~17ms avg inter-frame, not blocked by Byfron. Captures only the Roblox window (cleaner, smaller frames). **This is the recommended capture method.**
6. **WGC is 2x faster than MSS** (17ms vs 32ms) and provides window-targeted capture.
7. **End-to-end live pipeline: 37.9ms avg / 26 FPS** (preprocess + YOLO11n + postprocess, measured under WGC capture load). With capture overlapped: ~40ms / ~25 FPS.
8. **Keyboard input works via SendInput** (pynput, pydirectinput, ctypes). pydirectinput recommended (0.79ms avg tap latency).
9. **Mouse movement is blocked by Roblox.** All methods rejected. Bot uses **keyboard-only control** — WASD movement, arrow keys for camera, Space jump, E action.
10. **PyAutoGUI does not work with Roblox** — deprecated Win32 APIs rejected for all input types.
11. **YOLO11n pretrained COCO model detects Roblox characters as "person"** (0.60-0.75 confidence). Also detects "tv" for UI elements. Inference timing is content-independent (28.5ms real vs 28.3ms noise). Custom training needed for game-specific objects.
12. **Inference is ~5ms slower under concurrent WGC load** (26ms live vs 21ms isolated) — GPU/CPU resource contention. Acceptable.

### Optimization Roadmap

| Optimization | Expected Savings | Effort | Status |
|-------------|-----------------|--------|--------|
| Use WGC instead of DXcam/MSS | ~55ms (72ms -> 17ms capture) | Low | **VALIDATED — 60 FPS, not blocked** |
| Replace PIL with `cv2.resize` + contiguous | ~25ms (28ms -> 3ms) | Low | **VALIDATED 9.3x faster** |
| Run MobileNetV3 on CPU thread (parallel to GPU YOLO) | ~25ms (overlap) | Medium | Pending |
| Skip classification when not needed (detect-only frames) | Variable | Low | Pending |
| Pre-allocate numpy buffers (avoid alloc per frame) | ~2-5ms | Low | Tested: slower due to np.divide overhead |

### OpenCV Preprocessing Benchmark

Benchmark run: 2026-03-21. Script: `spikes/opencv-preprocess-benchmark/benchmark.py`

Preprocessing a 1920x1080 frame to 640x640 YOLO input (float32, NCHW):

| Method | Avg | P95 | Speedup vs PIL |
|--------|-----|-----|----------------|
| PIL Image.fromarray + resize + array | 28.4ms | 31.0ms | 1.0x (baseline) |
| cv2.resize + astype + transpose | 4.2ms | 5.0ms | 6.7x |
| **cv2.resize + ascontiguousarray** | **3.0ms** | **4.0ms** | **9.3x** |
| cv2.resize(dst=buf) + np.divide(out=buf) | 4.9ms | 5.5ms | 5.8x |

**Winner: `cv2.resize` + `np.ascontiguousarray` transpose** at 3.0ms avg.
The pre-allocated buffer approach was surprisingly slower due to `np.divide` overhead vs a simple float multiply.
All methods produce equivalent output shapes (1, 3, 640, 640) float32. Pixel-level differences between PIL and OpenCV (~0.16 mean) are due to different interpolation implementations and will not affect YOLO accuracy.

### YOLO11n on Real Roblox Frames

Benchmark run: 2026-03-21. Script: `spikes/yolo-roblox-detection/detect_test.py`

#### Inference timing: real content vs random noise

| Input | Avg | Median | P95 | FPS |
|-------|-----|--------|-----|-----|
| Real Roblox frame | 28.5ms | 28.7ms | 31.0ms | 35 |
| Random noise | 28.3ms | 28.6ms | 30.6ms | 35 |
| **Difference** | **0.17ms** | — | — | — |

Inference time is content-independent. The ~28ms avg (vs 21ms isolated) reflects GPU contention with concurrent WGC capture.

#### Live pipeline (WGC capture -> preprocess -> infer -> postprocess)

| Stage | Avg | Median | P95 | FPS |
|-------|-----|--------|-----|-----|
| Preprocess (cv2.resize + NCHW) | 8.7ms | 8.6ms | 10.1ms | 115 |
| YOLO11n inference (DirectML) | 26.0ms | 25.8ms | 28.4ms | 38 |
| Post-process (NMS + decode) | 3.2ms | 3.1ms | 4.3ms | 312 |
| **Full processing pipeline** | **37.9ms** | 37.9ms | 40.7ms | **26** |

#### COCO class detections in Roblox (50 live frames)

| Class | Occurrences | Notes |
|-------|-------------|-------|
| **person** | 168 | Roblox characters detected reliably (0.60-0.75 conf) |
| tv | 16 | Likely in-game signs or UI elements |

The pretrained COCO model provides a usable baseline for character detection. Custom training on Roblox-specific objects (items, NPCs, obstacles, UI) would be needed for full game awareness.

### Input Simulation

Benchmark run: 2026-03-21. Script: `spikes/input-simulation/input_test.py`

| Method | API | Avg Tap Latency | Roblox Keyboard | Roblox Mouse |
|--------|-----|----------------|-----------------|--------------|
| ctypes SendInput | Raw SendInput (scan codes) | **0.58ms** | Yes | No |
| pyautogui | keybd_event (deprecated) | 0.63ms | **No** | No |
| pynput | SendInput (VK codes) | 0.67ms | Yes | No |
| pydirectinput | SendInput (DI scan codes) | 0.79ms | Yes | No |

All working methods are sub-1ms — input latency is negligible in the pipeline.

### Behavior Tree (py_trees)

Benchmark run: 2026-03-21. Script: `spikes/behavior-tree/bt_spike.py`

Library: py_trees 2.4.0. Tree: 10 nodes (Selector -> 2 Sequences + 1 Action).

#### Correctness: 5/5 scenarios passed

| Scenario | Expected Action | Result |
|----------|----------------|--------|
| No detections | Wander (W + rotate) | PASS |
| Person far left | Navigate + rotate left | PASS |
| Person far right | Navigate + rotate right | PASS |
| Person centered, far | Navigate forward | PASS |
| Person centered, close | Steal (press E) | PASS |

#### Tick latency (isolated, 1000 iterations)

| Scenario | Avg | Median | P95 | Ticks/sec |
|----------|-----|--------|-----|-----------|
| No detections | 0.077ms | 0.067ms | 0.111ms | 12,958 |
| 1 detection | 0.087ms | 0.078ms | 0.128ms | 11,538 |
| 5 detections | 0.086ms | 0.079ms | 0.125ms | 11,661 |
| Alternating states | 0.086ms | 0.082ms | 0.127ms | 11,695 |

#### Live pipeline (WGC + YOLO + BT, 50 frames)

| Metric | Avg | Median | P95 | FPS |
|--------|-----|--------|-----|-----|
| Full pipeline (preprocess + YOLO + BT) | 37.6ms | 37.4ms | 41.0ms | 27 |
| BT tick only | 0.26ms | 0.23ms | 0.47ms | 3,825 |

Action distribution: 100% "Navigate to person" — YOLO detected persons in every frame, BT correctly chose the approach behavior.

BT tick overhead is **0.26ms** — negligible in the ~38ms pipeline. py_trees is confirmed suitable.

---

## Sources

- [pyrobloxbot - Roblox bot library](https://github.com/Mews/pyrobloxbot)
- [DXcam - High-performance screen capture](https://github.com/ra1nty/DXcam)
- [BetterCam - DXcam fork](https://github.com/RootKit-Org/BetterCam)
- [rblx-open-cloud - Open Cloud Python wrapper](https://github.com/treeben77/rblx-open-cloud)
- [ro.py - Roblox API wrapper](https://devforum.roblox.com/t/use-python-to-interact-with-the-roblox-api-with-ropy/1006465)
- [Roblox HttpService documentation](https://create.roblox.com/docs/reference/engine/classes/HttpService)
- [Roblox-Python communication tutorial](https://devforum.roblox.com/t/easy-communication-between-roblox-server-and-python/1091452)
- [AI game bot with PyTorch & EfficientNet](https://www.akshaymakes.com/blogs/pytorch)
- [Game bot with OpenCV](https://learncodebygaming.com/blog/how-to-build-a-bot-with-opencv)
- [Python game bot article (Medium)](https://medium.com/codrift/i-built-a-python-bot-that-plays-video-games-better-than-me-e7e390a868f9)
- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [End-to-End AI for NVIDIA PCs: ONNX and DirectML (NVIDIA Blog)](https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-onnx-and-directml/)
- [PyTorch Maxwell/Pascal support removal (GitHub issue)](https://github.com/pytorch/pytorch/issues/157517)
- [PyTorch GPU compute capability support](https://discuss.pytorch.org/t/gpu-compute-capability-support-for-each-pytorch-version/62434)
- [Ultralytics YOLO hardware requirements](https://github.com/ultralytics/ultralytics/issues/4106)
- [AI hardware requirements guide](https://localaimaster.com/blog/ai-hardware-requirements-2025-complete-guide)
