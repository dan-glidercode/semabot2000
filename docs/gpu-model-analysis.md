# GPU & AI Model Analysis for SeMaBot2000

## Hardware: NVIDIA GeForce GTX 960M

| Spec | Value |
|------|-------|
| Architecture | Maxwell (GM107) |
| VRAM | 4 GB GDDR5 |
| CUDA Compute Capability | 5.0 (sm_50) |
| CUDA Cores | 640 |
| Driver | 511.23 |
| CUDA Version | 11.6 |
| DirectX | 12 (feature level 12_1) |

---

## Inference Runtime Options

The GPU's Maxwell architecture limits which runtimes we can use. Here are the viable options ranked by practicality:

### 1. ONNX Runtime + DirectML (Primary Choice)

DirectML uses DirectX 12 instead of CUDA, bypassing all compute capability restrictions.

```
pip install onnxruntime-directml
```

- Works on any DX12 GPU including Maxwell
- No CUDA version dependency
- All models must be in ONNX format
- Slightly slower than native CUDA but far easier to set up

**Note:** DirectML is in maintenance mode at Microsoft. The successor is **WinML** (Windows 11 24H2+), which auto-selects the best backend. For now, `onnxruntime-directml` remains fully functional and is the pragmatic choice.

### 2. PyTorch 2.5.x + CUDA 12.1 (Fallback)

Last PyTorch version with Maxwell sm_50 support:

```
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
```

- Needed for training custom models locally
- Access to full Ultralytics YOLO ecosystem
- Locked to older PyTorch — no newer features

### 3. ONNX Runtime + CUDA 11.x (Alternative)

```
pip install onnxruntime-gpu  # ensure CUDA 11.x compatible version
```

- Slightly faster than DirectML on NVIDIA hardware
- Requires matching CUDA 11.6 toolkit installation
- Good option if CUDA is already configured

### 4. CPU Inference (For non-real-time tasks)

```
pip install onnxruntime
```

- No GPU needed, no compatibility issues
- Viable for OCR, LLM reasoning, and other non-latency-critical tasks

---

## Object Detection Models

These are the core models for understanding what's happening on screen.

### YOLO Family

| Model | Params | Size (ONNX) | FLOPs | mAP@50-95 | Expected FPS (960M) | VRAM |
|-------|--------|-------------|-------|-----------|---------------------|------|
| **YOLO11n (nano)** | 2.6M | ~5 MB | 6.5G | 39.5% | ~20-40 FPS | ~100-200 MB |
| **YOLO11s (small)** | 9.4M | ~19 MB | 21.5G | 47.0% | ~10-25 FPS | ~200-400 MB |
| **YOLOv8n (nano)** | 3.2M | ~6 MB | 8.7G | 37.3% | ~20-40 FPS | ~100-200 MB |
| **YOLOv8s (small)** | 11.2M | ~22 MB | 28.6G | 44.9% | ~10-20 FPS | ~200-400 MB |
| **YOLOv8m (medium)** | 25.9M | ~52 MB | 78.9G | 50.2% | ~5-10 FPS | ~500-800 MB |

**Recommendation:** **YOLO11n** — fewer parameters than YOLOv8n, better accuracy (39.5 vs 37.3 mAP), and optimized for ONNX export. At 640x640 input, expect 20-40 FPS on the 960M via DirectML, which is sufficient for the bot loop.

**Fallback:** If accuracy is insufficient, step up to YOLO11s. Avoid medium+ models — they'll eat too much VRAM and drop below real-time FPS.

### Ultra-Lightweight Alternatives

| Model | Size | FLOPs | mAP | Speed | Use Case |
|-------|------|-------|-----|-------|----------|
| **NanoDet-Plus-m** | 980 KB (int8) | 0.9G | ~30% | ~950+ FPS (CPU) | Extremely fast, low accuracy |
| **PicoDet-S** | ~2 MB | 0.7G | ~30% | ~1000+ FPS (CPU) | Similar to NanoDet |

These are viable if YOLO is too heavy, but their lower accuracy makes them better suited as a secondary fast-check detector rather than the primary model.

### RT-DETR (Transformer-based)

| Model | Params | mAP | Notes |
|-------|--------|-----|-------|
| **RT-DETR-L** | 32M | 53.1% | Transformer architecture, no NMS needed |
| **RT-DETR-R18** | ~20M | ~46% | Lighter variant with ResNet-18 backbone |

RT-DETR offers higher accuracy but is heavier. The smaller R18 variant might work but YOLO11n is a safer bet for our VRAM budget.

---

## Image Classification Models

For game state recognition (menus, inventory, combat, exploration, etc.).

| Model | Size (ONNX) | Params | Top-1 Acc | Inference | VRAM |
|-------|-------------|--------|-----------|-----------|------|
| **MobileNetV3-Small** | ~6 MB | 2.5M | 67.7% | <5ms | ~30-50 MB |
| **MobileNetV3-Large** | ~17 MB | 5.4M | 75.2% | <8ms | ~50-80 MB |
| **EfficientNet-Lite0** | ~17 MB | 4.7M | 75.1% | <10ms | ~50-80 MB |
| **EfficientNet-Lite4** | ~50 MB | 13M | 80.4% | <30ms | ~100-150 MB |
| **SqueezeNet 1.1** | ~5 MB | 1.2M | 58.2% | <3ms | ~20-40 MB |

**Recommendation:** **MobileNetV3-Large** — best accuracy-to-speed ratio. Train it on screenshots of different game states (menu, gameplay, inventory, death screen, etc.) and it classifies in under 8ms.

**Alternative:** EfficientNet-Lite0 if slightly higher accuracy is needed at similar cost.

---

## OCR (Reading In-Game Text)

For health bars, scores, chat messages, item names, and UI text.

| Engine | GPU VRAM | Speed (GPU) | Accuracy | Install |
|--------|----------|-------------|----------|---------|
| **Tesseract** | 0 (CPU only) | ~50-100ms | Good for clean text | `pip install pytesseract` + system install |
| **EasyOCR** | ~200 MB | ~30-50ms | Better for scene text | `pip install easyocr` |
| **PaddleOCR** | ~1.2 GB | ~80ms (12.7 FPS) | Best overall accuracy | `pip install paddleocr` |

**Recommendation:** **Tesseract** for clean UI text (numbers, labels) — zero VRAM usage, and game UI text is typically clean/high-contrast. Use **EasyOCR** only if Tesseract struggles with stylized game fonts. Avoid PaddleOCR — 1.2 GB VRAM is too expensive alongside YOLO.

**Optimization:** Crop small regions of interest (health bar, score area) before OCR rather than scanning the full screen. This dramatically improves speed.

---

## Template Matching (OpenCV)

Zero VRAM, CPU-based, highly effective for static UI elements.

| Method | Speed | Use Case |
|--------|-------|----------|
| `cv2.matchTemplate` | <1ms per template | Finding fixed UI icons, buttons, indicators |
| Color histogram analysis | <1ms | Detecting health bar fill level by color ratio |
| Contour detection | ~2-5ms | Finding UI boundaries and regions |
| Pixel sampling | <0.1ms | Reading specific pixel colors at known positions |

**Recommendation:** Use template matching extensively for **static UI elements** (minimap icons, health bars, inventory slots). It's essentially free and extremely reliable for elements that don't change position or appearance. Reserve YOLO for dynamic elements (players, NPCs, items in the 3D world).

---

## LLM / Reasoning Models

For high-level decision-making, not real-time frame processing.

| Model | Size | VRAM | Speed | Feasibility |
|-------|------|------|-------|-------------|
| **TinyLlama 1.1B (Q4_K_M)** | ~670 MB | ~1-1.5 GB | ~60 tok/s | Possible — leaves room for YOLO |
| **SmolLM2 1.7B (Q4_K_M)** | ~1 GB | ~1.5-2 GB | ~30-40 tok/s | Tight — competes with YOLO for VRAM |
| **Phi-3.5 Mini 3.8B (Q3)** | ~2 GB | ~3-3.5 GB | ~10-15 tok/s | Borderline — no room for anything else |
| **Claude API** | 0 (remote) | 0 | ~50-100ms network | Best reasoning, requires internet |

**Recommendation:** **Claude API** for complex reasoning (strategy, chat responses, goal planning). No local VRAM cost, far superior reasoning quality. Use local LLMs only if offline operation is required, in which case TinyLlama 1.1B Q4 is the only practical choice alongside YOLO.

---

## VRAM Budget

With 4 GB total, here's the realistic allocation:

### Configuration A: Vision-Focused (Recommended)

| Component | VRAM | Notes |
|-----------|------|-------|
| Windows / Display driver | ~300-500 MB | Always reserved |
| DXcam capture buffer | ~50-100 MB | Screen frames |
| YOLO11n (DirectML) | ~100-200 MB | Primary object detection |
| MobileNetV3-Large | ~50-80 MB | Game state classification |
| EasyOCR (if needed) | ~200 MB | Scene text reading |
| **Headroom** | **~2.5-3 GB free** | Comfortable margin |

### Configuration B: Vision + Local LLM

| Component | VRAM | Notes |
|-----------|------|-------|
| Windows / Display driver | ~300-500 MB | Always reserved |
| DXcam capture buffer | ~50-100 MB | Screen frames |
| YOLO11n (DirectML) | ~100-200 MB | Object detection |
| TinyLlama 1.1B Q4 | ~1-1.5 GB | Local reasoning |
| **Headroom** | **~1.5-2 GB free** | Tighter but workable |

### Configuration C: Maximum Vision

| Component | VRAM | Notes |
|-----------|------|-------|
| Windows / Display driver | ~300-500 MB | Always reserved |
| DXcam capture buffer | ~50-100 MB | Screen frames |
| YOLO11s (DirectML) | ~200-400 MB | Better detection accuracy |
| MobileNetV3-Large | ~50-80 MB | Game state classification |
| Tesseract OCR | 0 MB | CPU-based text reading |
| OpenCV templates | 0 MB | CPU-based UI detection |
| **Headroom** | **~2.5-3 GB free** | Comfortable |

---

## Recommended Model Stack

```
REAL-TIME PIPELINE (every frame, <50ms total):
  ├── DXcam                    → Screen capture         (~4ms, minimal VRAM)
  ├── YOLO11n (ONNX+DirectML) → Object detection        (~15-30ms, ~150 MB)
  ├── MobileNetV3-Large (ONNX) → Game state classify    (~5-8ms, ~60 MB)
  ├── OpenCV templates         → UI element matching     (~1-2ms, 0 VRAM)
  └── Tesseract (targeted ROI) → Read key text/numbers   (~10-20ms, 0 VRAM)

NON-REAL-TIME (on demand, async):
  ├── Claude API               → Strategy/reasoning      (~100ms+, 0 VRAM)
  └── EasyOCR (fallback)       → Complex text scenes     (~30-50ms, ~200 MB)
```

Total estimated VRAM for real-time pipeline: **~500-800 MB** (well within 4 GB budget)

---

## Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DirectML slower than expected on Maxwell | Lower FPS | Fall back to ONNX+CUDA 11.x; reduce input resolution to 416x416 |
| YOLO11n accuracy too low for game | Missed detections | Train custom model on Roblox screenshots; or upgrade to YOLO11s |
| PyTorch 2.5.x missing needed features | Can't train locally | Train on cloud (Colab/Kaggle free GPU) and export ONNX |
| Driver 511.23 too old for DirectML | Runtime errors | Update NVIDIA driver to latest Game Ready release |
| Roblox anti-cheat detects input sim | Bot blocked | Use human-like input timing with randomized delays |

---

## Next Steps

1. **Update NVIDIA driver** to latest version for best DirectML/DX12 compatibility
2. **Install ONNX Runtime DirectML** and run a benchmark with YOLO11n on the 960M
3. **Capture sample Roblox frames** with DXcam to verify capture pipeline
4. **Benchmark the full loop** (capture → detect → classify → act) to get real latency numbers
5. Based on benchmarks, decide between Configuration A, B, or C

---

## Sources

- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/models/yolov8/)
- [YOLO11 vs YOLOv8 comparison](https://docs.ultralytics.com/compare/yolo11-vs-yolov8/)
- [NanoDet-Plus](https://github.com/RangiLyu/nanodet)
- [PicoDet](https://medium.com/axinc-ai/picodet-fast-object-detection-model-optimized-for-mobile-cpus-17e7aa84589b)
- [Best object detection models 2025 (Ultralytics)](https://www.ultralytics.com/blog/the-best-object-detection-models-of-2025)
- [Best object detection models 2025 (Roboflow)](https://blog.roboflow.com/best-object-detection-models/)
- [EfficientNet-Lite (TensorFlow blog)](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)
- [EfficientNet-Lite4 ONNX model](https://github.com/onnx/models/tree/main/validated/vision/classification/efficientnet-lite4)
- [MobileNet ONNX model](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
- [OCR comparison: Tesseract vs EasyOCR vs PaddleOCR](https://toon-beerten.medium.com/ocr-comparison-tesseract-versus-easyocr-vs-paddleocr-vs-mmocr-a362d9c79e66)
- [ONNX Runtime DirectML docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [DirectML GitHub (maintenance mode notice)](https://github.com/microsoft/DirectML)
- [WinML with ONNX (AMD guide)](https://www.amd.com/en/developer/resources/technical-articles/2026/simplifying-onnx-deployment-with-winml.html)
- [PyTorch Maxwell support removal](https://github.com/pytorch/pytorch/issues/157517)
- [AI hardware requirements guide](https://localaimaster.com/blog/ai-hardware-requirements-2025-complete-guide)
- [Lightweight CNN comparison](https://arxiv.org/html/2505.03303v1)
