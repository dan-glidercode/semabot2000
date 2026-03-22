"""
DirectML Benchmark for GTX 960M (Maxwell, 4GB VRAM)
====================================================
Tests ONNX Runtime + DirectML inference performance with models
planned for the SeMaBot2000 real-time pipeline.

Benchmarks:
  1. YOLO11n  — object detection (primary model)
  2. MobileNetV3-Large — image classification
  3. DXcam — screen capture throughput
  4. Full pipeline — capture + detect + classify end-to-end

Run: python benchmark.py
"""

import argparse
import os
import sys
import time
import statistics
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_result(label: str, times_ms: list[float]):
    avg = statistics.mean(times_ms)
    med = statistics.median(times_ms)
    p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]
    mn = min(times_ms)
    mx = max(times_ms)
    fps = 1000.0 / avg if avg > 0 else 0
    print(f"  {label}")
    print(f"    avg: {avg:7.2f}ms  |  median: {med:7.2f}ms  |  p95: {p95:7.2f}ms")
    print(f"    min: {mn:7.2f}ms  |  max:    {mx:7.2f}ms  |  ~FPS: {fps:.1f}")


def warmup_session(session, input_name: str, dummy_input: np.ndarray, n: int = 5):
    """Run a few warmup inferences to let DirectML compile/optimize."""
    for _ in range(n):
        session.run(None, {input_name: dummy_input})


# ---------------------------------------------------------------------------
# 1. Export / download YOLO11n ONNX
# ---------------------------------------------------------------------------

def get_yolo11n_onnx(models_dir: Path) -> Path:
    onnx_path = models_dir / "yolo11n.onnx"
    if onnx_path.exists():
        print(f"  Found cached: {onnx_path}")
        return onnx_path

    print("  Exporting YOLO11n to ONNX via ultralytics (CPU, one-time)...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        model.export(format="onnx", imgsz=640, opset=17, simplify=True)
        # ultralytics exports next to the .pt file; find and move it
        exported = Path("yolo11n.onnx")
        if exported.exists():
            exported.rename(onnx_path)
        else:
            # check in ultralytics default location
            for candidate in [Path("runs/detect/train/weights/yolo11n.onnx"),
                              Path("yolo11n.onnx")]:
                if candidate.exists():
                    candidate.rename(onnx_path)
                    break
        # clean up downloaded .pt if desired
        pt_file = Path("yolo11n.pt")
        if pt_file.exists():
            pt_file.rename(models_dir / "yolo11n.pt")
    except Exception as e:
        print(f"  ERROR exporting YOLO11n: {e}")
        print("  Make sure ultralytics is installed: pip install ultralytics")
        sys.exit(1)

    if not onnx_path.exists():
        print(f"  ERROR: expected ONNX at {onnx_path} but not found")
        sys.exit(1)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  Exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# 2. Export MobileNetV3-Large ONNX
# ---------------------------------------------------------------------------

def get_mobilenetv3_onnx(models_dir: Path) -> Path:
    onnx_path = models_dir / "mobilenetv3_large.onnx"
    if onnx_path.exists():
        print(f"  Found cached: {onnx_path}")
        return onnx_path

    print("  Exporting MobileNetV3-Large to ONNX via torchvision (CPU, one-time)...")
    try:
        import torch
        import torchvision.models as models

        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except Exception as e:
        print(f"  ERROR exporting MobileNetV3: {e}")
        print("  Make sure torch and torchvision are installed.")
        sys.exit(1)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  Exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# 3. Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_onnx(onnx_path: Path, input_shape: tuple, provider: str,
                   n_warmup: int = 10, n_runs: int = 100) -> list[float]:
    """Run inference benchmark and return list of per-run times in ms."""
    import onnxruntime as ort

    providers = [provider]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        print(f"    FAILED to create session with {provider}: {e}")
        return []

    active_providers = session.get_providers()
    print(f"    Active providers: {active_providers}")

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    # Build dummy input matching expected shape
    dummy = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    warmup_session(session, input_name, dummy, n=n_warmup)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return times


def benchmark_dxcam(n_runs: int = 200) -> list[float]:
    """Benchmark DXcam screen capture speed."""
    try:
        import dxcam
    except ImportError:
        print("    dxcam not installed, skipping.")
        return []

    camera = dxcam.create()
    # Warmup — grab a few frames
    for _ in range(10):
        frame = camera.grab()
        if frame is None:
            time.sleep(0.01)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        frame = camera.grab()
        t1 = time.perf_counter()
        if frame is not None:
            times.append((t1 - t0) * 1000)

    if times and frame is not None:
        print(f"    Captured frame shape: {frame.shape}, dtype: {frame.dtype}")

    camera.stop() if hasattr(camera, 'stop') else None
    del camera
    return times


def benchmark_full_pipeline(yolo_path: Path, mobilenet_path: Path,
                            n_runs: int = 50) -> list[float]:
    """Simulate the full bot loop: capture → detect → classify → (act)."""
    import onnxruntime as ort

    try:
        import dxcam
        camera = dxcam.create()
    except ImportError:
        print("    dxcam not available — using synthetic frames.")
        camera = None

    # Create sessions
    dml = "DmlExecutionProvider"
    yolo_sess = ort.InferenceSession(str(yolo_path), providers=[dml])
    yolo_input = yolo_sess.get_inputs()[0].name

    mobilenet_sess = ort.InferenceSession(str(mobilenet_path), providers=[dml])
    mobilenet_input = mobilenet_sess.get_inputs()[0].name

    # Warmup
    yolo_dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    mobilenet_dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
    warmup_session(yolo_sess, yolo_input, yolo_dummy)
    warmup_session(mobilenet_sess, mobilenet_input, mobilenet_dummy)

    times = []
    capture_times = []
    detect_times = []
    classify_times = []

    for _ in range(n_runs):
        t_total_start = time.perf_counter()

        # CAPTURE
        t0 = time.perf_counter()
        if camera:
            frame = camera.grab()
            if frame is None:
                frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        t_capture = (time.perf_counter() - t0) * 1000

        # PREPROCESS for YOLO (resize + normalize + CHW + batch)
        t0 = time.perf_counter()
        from PIL import Image
        img = Image.fromarray(frame).resize((640, 640))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, 0)          # add batch dim

        # DETECT
        yolo_sess.run(None, {yolo_input: img_np})
        t_detect = (time.perf_counter() - t0) * 1000

        # PREPROCESS for classifier (resize to 224)
        t0 = time.perf_counter()
        cls_img = Image.fromarray(frame).resize((224, 224))
        cls_np = np.array(cls_img, dtype=np.float32) / 255.0
        cls_np = np.transpose(cls_np, (2, 0, 1))
        cls_np = np.expand_dims(cls_np, 0)

        # CLASSIFY
        mobilenet_sess.run(None, {mobilenet_input: cls_np})
        t_classify = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000

        capture_times.append(t_capture)
        detect_times.append(t_detect)
        classify_times.append(t_classify)
        times.append(t_total)

    if camera:
        camera.stop() if hasattr(camera, 'stop') else None
        del camera

    print_result("Capture", capture_times)
    print_result("YOLO11n detect (incl. preprocess)", detect_times)
    print_result("MobileNetV3 classify (incl. preprocess)", classify_times)
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DirectML benchmark for SeMaBot2000")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed runs per benchmark")
    parser.add_argument("--skip-dxcam", action="store_true", help="Skip DXcam benchmark")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip full pipeline benchmark")
    args = parser.parse_args()

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    # System info
    print_header("SYSTEM INFO")
    import onnxruntime as ort
    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Available providers:  {ort.get_available_providers()}")
    print(f"  Python:               {sys.version.split()[0]}")
    print(f"  NumPy:                {np.__version__}")

    has_dml = "DmlExecutionProvider" in ort.get_available_providers()
    if not has_dml:
        print("\n  WARNING: DmlExecutionProvider not available!")
        print("  Install: pip install onnxruntime-directml")
        print("  Falling back to CPU-only benchmarks.\n")

    # Export models
    print_header("MODEL PREPARATION")
    yolo_path = get_yolo11n_onnx(models_dir)
    mobilenet_path = get_mobilenetv3_onnx(models_dir)

    # --- YOLO11n benchmarks ---
    yolo_input_shape = (1, 3, 640, 640)

    if has_dml:
        print_header("YOLO11n — DirectML (GPU)")
        times = benchmark_onnx(yolo_path, yolo_input_shape, "DmlExecutionProvider",
                               n_runs=args.runs)
        if times:
            print_result("YOLO11n @ 640x640 (DirectML)", times)

    print_header("YOLO11n — CPU")
    times = benchmark_onnx(yolo_path, yolo_input_shape, "CPUExecutionProvider",
                           n_runs=args.runs)
    if times:
        print_result("YOLO11n @ 640x640 (CPU)", times)

    # Note: 416x416 test skipped — model was exported with fixed 640x640 dims.
    # Re-export with dynamic_axes if reduced resolution is needed later.

    # --- MobileNetV3 benchmarks ---
    mobilenet_shape = (1, 3, 224, 224)

    if has_dml:
        print_header("MobileNetV3-Large — DirectML (GPU)")
        times = benchmark_onnx(mobilenet_path, mobilenet_shape, "DmlExecutionProvider",
                               n_runs=args.runs)
        if times:
            print_result("MobileNetV3-Large @ 224x224 (DirectML)", times)

    print_header("MobileNetV3-Large — CPU")
    times = benchmark_onnx(mobilenet_path, mobilenet_shape, "CPUExecutionProvider",
                           n_runs=args.runs)
    if times:
        print_result("MobileNetV3-Large @ 224x224 (CPU)", times)

    # --- DXcam benchmark ---
    if not args.skip_dxcam:
        print_header("DXcam — Screen Capture")
        times = benchmark_dxcam(n_runs=args.runs * 2)
        if times:
            print_result("Screen capture (full screen)", times)
        else:
            print("    No frames captured (is a display connected?)")

    # --- Full pipeline ---
    if not args.skip_pipeline and has_dml:
        print_header("FULL PIPELINE — Capture → Detect → Classify")
        times = benchmark_full_pipeline(yolo_path, mobilenet_path, n_runs=min(args.runs, 50))
        if times:
            print_result("Full pipeline (total)", times)

    # --- Summary ---
    print_header("BENCHMARK COMPLETE")
    print("  Results above show real latency on this GPU.")
    print("  Key question: Is DirectML YOLO11n @ 640x640 under 50ms?")
    print("  If not, options:")
    print("    - Use 416x416 input resolution")
    print("    - Try onnxruntime-gpu with CUDA 11.x instead")
    print("    - Use NanoDet/PicoDet for faster (but less accurate) detection")
    print()


if __name__ == "__main__":
    main()
