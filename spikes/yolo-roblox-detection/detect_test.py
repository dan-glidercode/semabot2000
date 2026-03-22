"""
YOLO11n Detection on Real Roblox Frames
=========================================
Tests YOLO11n (ONNX + DirectML) on actual Roblox screenshots captured
via WGC to answer:

  1. Does inference time change with real game content vs random noise?
  2. What does the pretrained COCO model detect in Roblox?
  3. What does the full capture -> preprocess -> infer pipeline measure?

Uses saved captures from the WGC spike, plus live capture if Roblox is running.

Run: python detect_test.py
"""

import sys
import time
import statistics
import threading
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

MODELS_DIR = Path(__file__).parent / ".." / "directml-benchmark" / "models"
OUTPUT_DIR = Path(__file__).parent / "detections"
OUTPUT_DIR.mkdir(exist_ok=True)

WARMUP = 10
ITERATIONS = 50

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


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
    print(f"    avg: {avg:7.2f}ms  |  median: {med:7.2f}ms  |  p95: {p95:7.2f}ms")
    print(f"    min: {mn:7.2f}ms  |  max:    {mx:7.2f}ms  |  ~FPS: {fps:.0f}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 frame -> (1, 3, 640, 640) float32 NCHW blob."""
    resized = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    blob *= (1.0 / 255.0)
    return blob[np.newaxis]


# ---------------------------------------------------------------------------
# Post-processing: parse YOLO11n raw output into detections
# ---------------------------------------------------------------------------

def postprocess(output: np.ndarray, conf_threshold: float = 0.25,
                orig_w: int = 640, orig_h: int = 640):
    """
    Parse YOLO11n output tensor (1, 84, 8400) into list of detections.
    Each detection: (class_id, class_name, confidence, x1, y1, x2, y2)
    Coordinates are in 640x640 space.
    """
    # output shape: (1, 84, 8400) -> transpose to (8400, 84)
    preds = output[0].T  # (8400, 84)

    # First 4 values: cx, cy, w, h
    # Remaining 80: class confidences
    boxes = preds[:, :4]
    class_scores = preds[:, 4:]

    # Get max class confidence and class id for each detection
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Convert cx, cy, w, h to x1, y1, x2, y2
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Simple NMS (greedy, per-class)
    detections = []
    for i in range(len(scores)):
        detections.append({
            "class_id": int(class_ids[i]),
            "class_name": COCO_CLASSES[int(class_ids[i])] if int(class_ids[i]) < len(COCO_CLASSES) else f"class_{class_ids[i]}",
            "confidence": float(scores[i]),
            "bbox": (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Simple greedy NMS
    kept = []
    for det in detections:
        overlap = False
        for k in kept:
            if det["class_id"] == k["class_id"]:
                # IoU check
                bx1 = max(det["bbox"][0], k["bbox"][0])
                by1 = max(det["bbox"][1], k["bbox"][1])
                bx2 = min(det["bbox"][2], k["bbox"][2])
                by2 = min(det["bbox"][3], k["bbox"][3])
                inter = max(0, bx2 - bx1) * max(0, by2 - by1)
                area_a = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
                area_b = (k["bbox"][2] - k["bbox"][0]) * (k["bbox"][3] - k["bbox"][1])
                union = area_a + area_b - inter
                if union > 0 and inter / union > 0.5:
                    overlap = True
                    break
        if not overlap:
            kept.append(det)

    return kept


# ---------------------------------------------------------------------------
# Draw detections on frame
# ---------------------------------------------------------------------------

def draw_detections(frame_bgr: np.ndarray, detections: list,
                    input_size: int = 640) -> np.ndarray:
    """Draw bounding boxes on the original frame."""
    h, w = frame_bgr.shape[:2]
    scale_x = w / input_size
    scale_y = h / input_size
    out = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Scale back to original frame coordinates
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    return out


# ---------------------------------------------------------------------------
# Load YOLO11n ONNX session
# ---------------------------------------------------------------------------

def load_yolo_session() -> ort.InferenceSession:
    yolo_path = MODELS_DIR / "yolo11n.onnx"
    if not yolo_path.exists():
        print(f"  ERROR: YOLO11n model not found at {yolo_path}")
        print("  Run the directml-benchmark first to export the model.")
        sys.exit(1)

    providers = ort.get_available_providers()
    has_dml = "DmlExecutionProvider" in providers
    provider = "DmlExecutionProvider" if has_dml else "CPUExecutionProvider"
    print(f"  Provider: {provider}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(yolo_path), opts, providers=[provider])
    return session


# ---------------------------------------------------------------------------
# Test 1: Inference on saved WGC captures
# ---------------------------------------------------------------------------

def test_saved_frames(session: ort.InferenceSession):
    print_header("Test 1: YOLO11n on Saved Roblox Captures")

    # Look for captures from the WGC spike
    capture_dirs = [
        Path(__file__).parent / ".." / "wgc-capture-benchmark" / "captures",
        Path(__file__).parent / ".." / "dxcam-roblox-capture" / "captures",
    ]

    frames = []
    for cap_dir in capture_dirs:
        if cap_dir.exists():
            for img_path in sorted(cap_dir.glob("*.png"))[:5]:
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    frames.append((img_path.name, frame))

    if not frames:
        print("  No saved captures found. Skipping.")
        return

    print(f"  Found {len(frames)} saved frames")

    input_name = session.get_inputs()[0].name

    for name, frame in frames:
        print(f"\n  --- {name} ({frame.shape[1]}x{frame.shape[0]}) ---")

        # Preprocess
        blob = preprocess(frame)

        # Inference
        t0 = time.perf_counter()
        output = session.run(None, {input_name: blob})
        infer_ms = (time.perf_counter() - t0) * 1000
        print(f"  Inference: {infer_ms:.1f}ms")

        # Post-process
        detections = postprocess(output[0], conf_threshold=0.25)
        print(f"  Detections ({len(detections)}):")
        for det in detections[:10]:
            print(f"    {det['class_name']:15s}  conf={det['confidence']:.3f}  "
                  f"bbox=({det['bbox'][0]:.0f},{det['bbox'][1]:.0f},"
                  f"{det['bbox'][2]:.0f},{det['bbox'][3]:.0f})")

        # Save annotated frame
        annotated = draw_detections(frame, detections)
        out_path = OUTPUT_DIR / f"det_{name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Test 2: Inference timing (real frames vs random noise)
# ---------------------------------------------------------------------------

def test_inference_timing(session: ort.InferenceSession):
    print_header("Test 2: Inference Timing — Real Frames vs Random Noise")

    input_name = session.get_inputs()[0].name

    # Load a real frame
    capture_dirs = [
        Path(__file__).parent / ".." / "wgc-capture-benchmark" / "captures",
        Path(__file__).parent / ".." / "dxcam-roblox-capture" / "captures",
    ]

    real_frame = None
    for cap_dir in capture_dirs:
        if cap_dir.exists():
            for img_path in cap_dir.glob("*.png"):
                real_frame = cv2.imread(str(img_path))
                if real_frame is not None:
                    break
        if real_frame is not None:
            break

    if real_frame is None:
        print("  No saved captures found. Skipping comparison.")
        return

    real_blob = preprocess(real_frame)
    noise_blob = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # Warmup both
    for _ in range(WARMUP):
        session.run(None, {input_name: real_blob})
        session.run(None, {input_name: noise_blob})

    # Benchmark real frames
    real_times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        session.run(None, {input_name: real_blob})
        real_times.append((time.perf_counter() - t0) * 1000)

    print_result("YOLO11n on real Roblox frame", real_times)

    # Benchmark random noise
    noise_times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        session.run(None, {input_name: noise_blob})
        noise_times.append((time.perf_counter() - t0) * 1000)

    print_result("YOLO11n on random noise", noise_times)

    diff = abs(statistics.mean(real_times) - statistics.mean(noise_times))
    print(f"\n  Difference: {diff:.2f}ms (should be near zero)")


# ---------------------------------------------------------------------------
# Test 3: Live WGC capture -> detect pipeline
# ---------------------------------------------------------------------------

def test_live_pipeline(session: ort.InferenceSession):
    print_header("Test 3: Live WGC Capture -> Detect Pipeline")

    try:
        import win32gui
        from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    except ImportError:
        print("  windows-capture or pywin32 not installed. Skipping.")
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
        print("  Roblox not running. Skipping live pipeline test.")
        return

    print(f"  Targeting: \"{roblox[0]}\"")

    input_name = session.get_inputs()[0].name

    # Warmup the model
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(WARMUP):
        session.run(None, {input_name: dummy})

    # Collect frames and run full pipeline
    pipeline_times = []
    preprocess_times = []
    infer_times = []
    postprocess_times = []
    all_detections = []
    saved_frames = []
    lock = threading.Lock()
    done_event = threading.Event()
    frame_count = [0]
    total_needed = 10 + ITERATIONS  # 10 warmup

    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=roblox[0],
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        with lock:
            count = frame_count[0]
            frame_count[0] += 1

        if count >= total_needed:
            capture_control.stop()
            done_event.set()
            return

        t_total = time.perf_counter()

        # Preprocess
        t0 = time.perf_counter()
        img = frame.frame_buffer
        bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        blob = preprocess(bgr)
        t_preprocess = (time.perf_counter() - t0) * 1000

        # Inference
        t0 = time.perf_counter()
        output = session.run(None, {input_name: blob})
        t_infer = (time.perf_counter() - t0) * 1000

        # Post-process
        t0 = time.perf_counter()
        detections = postprocess(output[0], conf_threshold=0.25)
        t_post = (time.perf_counter() - t0) * 1000

        t_pipeline = (time.perf_counter() - t_total) * 1000

        if count >= 10:  # skip warmup
            with lock:
                preprocess_times.append(t_preprocess)
                infer_times.append(t_infer)
                postprocess_times.append(t_post)
                pipeline_times.append(t_pipeline)
                all_detections.append(detections)
                # Save first few annotated frames
                if len(saved_frames) < 5:
                    saved_frames.append((bgr.copy(), detections))

    @capture.event
    def on_closed():
        done_event.set()

    print(f"  Running {ITERATIONS} pipeline iterations...")

    capture_thread = threading.Thread(target=capture.start, daemon=True)
    capture_thread.start()
    done_event.wait(timeout=60)

    with lock:
        p_times = list(pipeline_times)
        pre_times = list(preprocess_times)
        inf_times = list(infer_times)
        post_times = list(postprocess_times)
        dets = list(all_detections)
        frames_to_save = list(saved_frames)

    if not p_times:
        print("  No frames processed.")
        return

    print_result("Preprocess (cv2.resize + NCHW)", pre_times)
    print_result("YOLO11n inference (DirectML)", inf_times)
    print_result("Post-process (NMS + decode)", post_times)
    print_result("Full pipeline (preprocess + infer + postprocess)", p_times)

    # Detection summary
    print(f"\n  Detection summary across {len(dets)} frames:")
    class_counts = {}
    for frame_dets in dets:
        for det in frame_dets:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

    if class_counts:
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:20s}: detected in {count} frames")
    else:
        print("    No COCO objects detected in any frame.")

    # Save annotated frames
    for i, (frame, detections) in enumerate(frames_to_save):
        annotated = draw_detections(frame, detections)
        out_path = OUTPUT_DIR / f"live_{i:02d}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved: {out_path.name} ({len(detections)} detections)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("YOLO11n Roblox Detection Spike")

    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  Providers: {ort.get_available_providers()}")

    session = load_yolo_session()

    test_saved_frames(session)
    test_inference_timing(session)
    test_live_pipeline(session)

    print_header("DONE")
    print(f"  Annotated frames saved to: {OUTPUT_DIR.resolve()}")
    print("  Check the images to see what COCO classes YOLO detects in Roblox.")
    print()


if __name__ == "__main__":
    main()
