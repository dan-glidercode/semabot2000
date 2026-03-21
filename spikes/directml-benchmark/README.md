# DirectML Benchmark Spike

Validates whether ONNX Runtime + DirectML can run our planned AI models
at real-time speeds on the GTX 960M (4 GB VRAM, Maxwell, sm_50).

## What it tests

| Benchmark | Model | Input | Goal |
|-----------|-------|-------|------|
| Object detection | YOLO11n | 640x640 | <30ms per frame |
| Object detection (reduced) | YOLO11n | 416x416 | <20ms per frame |
| Classification | MobileNetV3-Large | 224x224 | <10ms per frame |
| Screen capture | DXcam | Full screen | <5ms per grab |
| Full pipeline | All of the above | — | <50ms total |

Each benchmark runs DirectML (GPU) and CPU to compare.

## Setup

```bash
cd spikes/directml-benchmark
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: `ultralytics` pulls in PyTorch as a dependency (needed for one-time
ONNX export). If PyTorch CUDA doesn't work on Maxwell, the export still
runs on CPU — it only needs GPU for training, not export.

## Run

```bash
python benchmark.py
```

Options:
- `--runs 200` — more iterations for stable numbers (default: 100)
- `--skip-dxcam` — skip screen capture test (e.g. running headless)
- `--skip-pipeline` — skip the combined pipeline test

## Expected output

```
============================================================
  YOLO11n — DirectML (GPU)
============================================================
    Active providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
  YOLO11n @ 640x640 (DirectML)
    avg:   XX.XXms  |  median:   XX.XXms  |  p95:   XX.XXms
    min:   XX.XXms  |  max:      XX.XXms  |  ~FPS: XX.X
...
```

## Interpreting results

- **DirectML faster than CPU?** If yes, the GPU is accelerating inference.
  If not, the 960M may not benefit from DirectML and we should try CUDA 11.x.
- **YOLO11n @ 640 under 30ms?** Great — use 640x640 as default input size.
- **YOLO11n @ 640 over 50ms?** Try 416x416. If still too slow, consider
  NanoDet-Plus or running YOLO on CPU (which can be competitive for nano models).
- **Full pipeline under 50ms?** The bot can run at 20+ FPS — sufficient for
  real-time gameplay.

## Output files

After first run, exported models are cached in `models/`:
- `models/yolo11n.onnx`
- `models/mobilenetv3_large.onnx`
