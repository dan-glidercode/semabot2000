"""
Double-Buffer + Frame-Skip Optimization Spike
===============================================
Tests two optimizations for the SeMaBot2000 pipeline:

1. Double-buffered capture: WGC fills a buffer while the processing
   thread works on the previous frame. Eliminates the capture-wait
   bottleneck (26.7ms in e2e spike).

2. Frame-skip: if the new frame is identical (or very similar) to the
   previous frame, skip YOLO inference and reuse last detections.
   Saves ~26ms per skipped frame.

Benchmarks:
  A. Sequential baseline (current): capture-wait -> preprocess -> detect
  B. Double-buffered: capture overlapped with processing
  C. Double-buffered + frame-skip

Run from a regular terminal with Roblox visible:
  ..\\directml-benchmark\\.venv\\Scripts\\python.exe double_buffer_spike.py
"""

from __future__ import annotations

import hashlib
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

YOLO_MODEL = Path(__file__).parent / ".." / "directml-benchmark" / "models" / "yolo11n.onnx"
INPUT_SIZE = 640
DURATION_S = 5.0
CONFIDENCE = 0.35


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
    fps = 1000.0 / avg if avg > 0 else 0
    print(f"  {label}")
    print(f"    avg: {avg:7.1f}ms  median: {med:7.1f}ms  p95: {p95:7.1f}ms  FPS: {fps:.0f}")


# ---------------------------------------------------------------------------
# Shared: YOLO preprocess + detect
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    blob *= 1.0 / 255.0
    return blob[np.newaxis]


def detect(session: ort.InferenceSession, input_name: str, blob: np.ndarray) -> int:
    output = session.run(None, {input_name: blob})
    preds = output[0][0].T
    scores = preds[:, 4:].max(axis=1)
    return int((scores > CONFIDENCE).sum())


def frame_hash(frame: np.ndarray) -> str:
    """Fast hash of downsampled frame for change detection."""
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_NEAREST)
    return hashlib.md5(small.tobytes()).hexdigest()  # noqa: S324


# ---------------------------------------------------------------------------
# Find Roblox
# ---------------------------------------------------------------------------

def find_roblox() -> str | None:
    import win32gui

    results = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and "roblox" in title.lower():
                results.append(title)

    win32gui.EnumWindows(cb, None)
    return results[0] if results else None


# ---------------------------------------------------------------------------
# A. Sequential baseline (current pipeline)
# ---------------------------------------------------------------------------

def benchmark_sequential(
    session: ort.InferenceSession,
    input_name: str,
    window_title: str,
) -> list[float]:
    print_header("A. Sequential Baseline")
    print("  capture-wait -> preprocess -> detect (current pipeline)")

    from windows_capture import Frame, InternalCaptureControl, WindowsCapture

    latest_frame = [None]
    frame_lock = threading.Lock()
    frame_ready = threading.Event()
    running = threading.Event()
    running.set()

    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=window_title,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        if not running.is_set():
            capture_control.stop()
            return
        img = frame.frame_buffer
        bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        with frame_lock:
            latest_frame[0] = bgr.copy()
        frame_ready.set()

    @capture.event
    def on_closed():
        frame_ready.set()

    cap_thread = threading.Thread(target=capture.start, daemon=True)
    cap_thread.start()
    frame_ready.wait(timeout=5)

    times = []
    start = time.perf_counter()

    while (time.perf_counter() - start) < DURATION_S:
        t0 = time.perf_counter()

        # Wait for frame
        frame_ready.clear()
        frame_ready.wait(timeout=0.1)
        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            continue

        # Process
        blob = preprocess(frame)
        detect(session, input_name, blob)

        times.append((time.perf_counter() - t0) * 1000)

    running.clear()
    frame_ready.set()
    print_result("Sequential total", times)
    return times


# ---------------------------------------------------------------------------
# B. Double-buffered: capture thread continuously fills buffer,
#    processing thread grabs latest and works on it
# ---------------------------------------------------------------------------

def benchmark_double_buffer(
    session: ort.InferenceSession,
    input_name: str,
    window_title: str,
) -> list[float]:
    print_header("B. Double-Buffered")
    print("  Capture thread fills buffer; processing grabs latest")

    from windows_capture import Frame, InternalCaptureControl, WindowsCapture

    latest_frame = [None]
    frame_lock = threading.Lock()
    frame_ready = threading.Event()
    running = threading.Event()
    running.set()

    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=window_title,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        if not running.is_set():
            capture_control.stop()
            return
        img = frame.frame_buffer
        bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        with frame_lock:
            latest_frame[0] = bgr.copy()
        frame_ready.set()

    @capture.event
    def on_closed():
        frame_ready.set()

    cap_thread = threading.Thread(target=capture.start, daemon=True)
    cap_thread.start()
    frame_ready.wait(timeout=5)

    times = []
    start = time.perf_counter()

    while (time.perf_counter() - start) < DURATION_S:
        t0 = time.perf_counter()

        # Grab latest frame WITHOUT waiting — use whatever is available
        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            time.sleep(0.001)
            continue

        # Process (capture thread continues filling buffer in parallel)
        blob = preprocess(frame)
        detect(session, input_name, blob)

        times.append((time.perf_counter() - t0) * 1000)

    running.clear()
    frame_ready.set()
    print_result("Double-buffered total", times)
    return times


# ---------------------------------------------------------------------------
# C. Double-buffered + frame-skip
# ---------------------------------------------------------------------------

def benchmark_double_buffer_frameskip(
    session: ort.InferenceSession,
    input_name: str,
    window_title: str,
) -> list[float]:
    print_header("C. Double-Buffered + Frame-Skip")
    print("  Skip YOLO if frame unchanged (hash comparison)")

    from windows_capture import Frame, InternalCaptureControl, WindowsCapture

    latest_frame = [None]
    frame_lock = threading.Lock()
    frame_ready = threading.Event()
    running = threading.Event()
    running.set()

    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=window_title,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        if not running.is_set():
            capture_control.stop()
            return
        img = frame.frame_buffer
        bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        with frame_lock:
            latest_frame[0] = bgr.copy()
        frame_ready.set()

    @capture.event
    def on_closed():
        frame_ready.set()

    cap_thread = threading.Thread(target=capture.start, daemon=True)
    cap_thread.start()
    frame_ready.wait(timeout=5)

    times = []
    last_hash = ""
    skipped = 0
    processed = 0
    start = time.perf_counter()

    while (time.perf_counter() - start) < DURATION_S:
        t0 = time.perf_counter()

        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            time.sleep(0.001)
            continue

        # Check if frame changed
        h = frame_hash(frame)
        if h == last_hash:
            skipped += 1
            times.append((time.perf_counter() - t0) * 1000)
            continue

        last_hash = h
        processed += 1

        # Full processing
        blob = preprocess(frame)
        detect(session, input_name, blob)

        times.append((time.perf_counter() - t0) * 1000)

    running.clear()
    frame_ready.set()

    total = skipped + processed
    skip_pct = 100 * skipped / total if total > 0 else 0
    print(f"  Frames: {total} total, {processed} processed, {skipped} skipped ({skip_pct:.0f}%)")
    print_result("Double-buffered + skip total", times)
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("Double-Buffer + Frame-Skip Optimization Spike")

    window_title = find_roblox()
    if not window_title:
        print("  ERROR: Roblox not found. Join a game first.")
        sys.exit(1)
    print(f"  Window: \"{window_title}\"")

    if not YOLO_MODEL.exists():
        print(f"  ERROR: Model not found: {YOLO_MODEL}")
        sys.exit(1)

    print("  Loading YOLO11n (DirectML)...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    session = ort.InferenceSession(
        str(YOLO_MODEL), opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    print("  Warming up (10 inferences)...")
    dummy = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    for _ in range(10):
        session.run(None, {input_name: dummy})

    print(f"\n  Running each benchmark for {DURATION_S}s...")
    print("  Keep Roblox visible!\n")

    seq_times = benchmark_sequential(session, input_name, window_title)
    time.sleep(1)
    db_times = benchmark_double_buffer(session, input_name, window_title)
    time.sleep(1)
    dbfs_times = benchmark_double_buffer_frameskip(session, input_name, window_title)

    # Summary
    print_header("SUMMARY")
    results = [
        ("A. Sequential (baseline)", seq_times),
        ("B. Double-buffered", db_times),
        ("C. Double-buf + frame-skip", dbfs_times),
    ]

    print(f"  {'Method':<30s}  {'Avg':>8s}  {'Median':>8s}  {'FPS':>6s}  {'Speedup':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")

    baseline_avg = statistics.mean(seq_times) if seq_times else 1
    for label, times in results:
        if not times:
            print(f"  {label:<30s}      N/A       N/A     N/A       N/A")
            continue
        avg = statistics.mean(times)
        med = statistics.median(times)
        fps = 1000.0 / avg
        speedup = baseline_avg / avg
        print(f"  {label:<30s}  {avg:7.1f}ms  {med:7.1f}ms  {fps:5.0f}  {speedup:7.2f}x")

    print()


if __name__ == "__main__":
    main()
