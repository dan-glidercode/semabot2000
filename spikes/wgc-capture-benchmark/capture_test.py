"""
Windows Graphics Capture API Benchmark
=======================================
Tests the WGC API via the windows-capture library as an alternative to
MSS (GDI BitBlt) and DXcam (DXGI, blocked by Byfron).

WGC is the same API used by OBS Studio for window capture. It operates
at a different level than DXGI Desktop Duplication and may not be blocked
by Roblox's anti-cheat.

Tests:
  1. WGC window capture (target Roblox by window name)
  2. Frame latency and throughput
  3. Capture -> OpenCV preprocess pipeline
  4. Side-by-side comparison with MSS

Run from a regular terminal (not VS Code embedded):
  ..\\directml-benchmark\\.venv\\Scripts\\python.exe capture_test.py
"""

import sys
import time
import statistics
import threading
from pathlib import Path

import numpy as np
import cv2

OUTPUT_DIR = Path(__file__).parent / "captures"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_FRAMES = 150
WARMUP_FRAMES = 20


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


def validate_frame(frame: np.ndarray, label: str) -> bool:
    if frame is None:
        print(f"    {label}: FAIL - frame is None")
        return False
    mean_val = frame.mean()
    std_val = frame.std()
    print(f"    {label}: shape={frame.shape}, dtype={frame.dtype}, "
          f"mean={mean_val:.1f}, std={std_val:.1f}", end="")
    if mean_val < 5 and std_val < 2:
        print(" -> LIKELY BLACK")
        return False
    if std_val < 1:
        print(" -> LIKELY FROZEN")
        return False
    print(" -> OK")
    return True


def save_frame(frame: np.ndarray, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    cv2.imwrite(str(path), frame)
    size_kb = path.stat().st_size / 1024
    print(f"    Saved: {path.name} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Find Roblox window
# ---------------------------------------------------------------------------

def find_roblox_window():
    import win32gui
    results = []

    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and "roblox" in title.lower():
            rect = win32gui.GetWindowRect(hwnd)
            results.append({
                "hwnd": hwnd, "title": title,
                "left": rect[0], "top": rect[1],
                "right": rect[2], "bottom": rect[3],
                "width": rect[2] - rect[0], "height": rect[3] - rect[1],
            })

    win32gui.EnumWindows(enum_cb, None)
    if not results:
        return None
    results.sort(key=lambda w: w["width"] * w["height"], reverse=True)
    return results[0]


# ---------------------------------------------------------------------------
# Test 1: Windows Graphics Capture API
# ---------------------------------------------------------------------------

def test_wgc(window: dict):
    print_header("Windows Graphics Capture API (WGC)")

    from windows_capture import WindowsCapture, Frame, InternalCaptureControl

    frames = []
    timestamps = []
    total_needed = WARMUP_FRAMES + TARGET_FRAMES
    lock = threading.Lock()
    done_event = threading.Event()

    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=window["title"],
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        t = time.perf_counter()
        with lock:
            count = len(timestamps)

        if count >= total_needed:
            capture_control.stop()
            done_event.set()
            return

        # Get numpy array (BGRA) via frame_buffer
        img = frame.frame_buffer.copy()

        with lock:
            timestamps.append(t)
            # Keep a few frames for validation, discard rest to save memory
            if count < WARMUP_FRAMES + 5 or count == total_needed - 1:
                frames.append((count, img))

    @capture.event
    def on_closed():
        done_event.set()

    print(f"  Targeting window: \"{window['title']}\"")
    print(f"  Collecting {total_needed} frames ({WARMUP_FRAMES} warmup + {TARGET_FRAMES} timed)...")

    try:
        # start() blocks, so run in a thread
        capture_thread = threading.Thread(target=capture.start, daemon=True)
        capture_thread.start()

        # Wait for frames with timeout
        done_event.wait(timeout=30)

        with lock:
            num_frames = len(timestamps)

        if num_frames < WARMUP_FRAMES + 10:
            print(f"  WARNING: Only got {num_frames} frames (needed {total_needed})")
            if num_frames == 0:
                print("  WGC capture produced no frames. Possibly blocked.")
                return [], []

    except Exception as e:
        print(f"  FAILED: {e}")
        return [], []

    # Calculate inter-frame intervals (latency between consecutive frames)
    with lock:
        ts = list(timestamps)
        saved_frames = list(frames)

    intervals = []
    for i in range(WARMUP_FRAMES + 1, len(ts)):
        intervals.append((ts[i] - ts[i - 1]) * 1000)

    print(f"  Total frames received: {len(ts)}")
    print(f"  Timed intervals: {len(intervals)}")

    if intervals:
        print_result("WGC inter-frame interval", intervals)

    # Validate and save sample frames
    print("\n  Sample frames:")
    for idx, img in saved_frames[:3]:
        # BGRA -> BGR for saving/validation
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            bgr = img
        validate_frame(bgr, f"Frame #{idx}")
        save_frame(bgr, f"wgc_{idx:03d}")

    return intervals, saved_frames


# ---------------------------------------------------------------------------
# Test 2: MSS baseline (for comparison)
# ---------------------------------------------------------------------------

def test_mss(window: dict):
    print_header("MSS (GDI BitBlt) — Baseline Comparison")

    import mss

    monitor = {
        "left": max(0, window["left"]),
        "top": max(0, window["top"]),
        "width": window["width"],
        "height": window["height"],
    }

    with mss.mss() as sct:
        # Warmup
        for _ in range(WARMUP_FRAMES):
            sct.grab(monitor)

        times = []
        for _ in range(TARGET_FRAMES):
            t0 = time.perf_counter()
            shot = sct.grab(monitor)
            frame = np.array(shot)[:, :, :3]
            times.append((time.perf_counter() - t0) * 1000)

        print_result("MSS grab() + np.array", times)
        return times


# ---------------------------------------------------------------------------
# Test 3: WGC capture -> preprocess pipeline
# ---------------------------------------------------------------------------

def test_wgc_preprocess_pipeline(window: dict):
    print_header("WGC Capture -> Preprocess Pipeline")

    from windows_capture import WindowsCapture, Frame, InternalCaptureControl

    pipeline_times = []
    lock = threading.Lock()
    done_event = threading.Event()
    total_needed = WARMUP_FRAMES + TARGET_FRAMES

    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=window["title"],
    )

    frame_count = [0]

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        with lock:
            count = frame_count[0]
            frame_count[0] += 1

        if count >= total_needed:
            capture_control.stop()
            done_event.set()
            return

        t0 = time.perf_counter()

        # Full preprocess pipeline: WGC frame -> numpy -> BGR -> resize -> NCHW float32
        img = frame.frame_buffer
        bgr = img[:, :, :3] if img.shape[2] == 4 else img
        resized = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        blob *= (1.0 / 255.0)
        blob = blob[np.newaxis]

        elapsed = (time.perf_counter() - t0) * 1000

        if count >= WARMUP_FRAMES:
            with lock:
                pipeline_times.append(elapsed)

    @capture.event
    def on_closed():
        done_event.set()

    print(f"  Measuring frame-to-YOLO-input preprocessing time...")

    capture_thread = threading.Thread(target=capture.start, daemon=True)
    capture_thread.start()
    done_event.wait(timeout=30)

    with lock:
        times = list(pipeline_times)

    if times:
        print_result("WGC frame -> preprocess (in callback)", times)
        print(f"  Note: this measures only numpy + resize + normalize, not capture latency")
    else:
        print("  No frames preprocessed.")

    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("WGC vs MSS Capture Benchmark")

    window = find_roblox_window()
    if window is None:
        print("  ERROR: No Roblox window found!")
        print("  Make sure Roblox is running and you've joined a game.")
        sys.exit(1)

    print(f"  Window: \"{window['title']}\" ({window['width']}x{window['height']})")

    # Test WGC
    wgc_intervals, wgc_frames = test_wgc(window)

    # Test MSS
    mss_times = test_mss(window)

    # Test WGC preprocess
    wgc_preprocess_times = test_wgc_preprocess_pipeline(window)

    # Summary
    print_header("SUMMARY")
    print(f"  {'Method':<40s}  {'Avg':>8s}  {'P95':>8s}  {'FPS':>6s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*6}")

    if wgc_intervals:
        avg = statistics.mean(wgc_intervals)
        p95 = sorted(wgc_intervals)[int(len(wgc_intervals) * 0.95)]
        print(f"  {'WGC inter-frame interval':<40s}  {avg:7.1f}ms  {p95:7.1f}ms  {1000/avg:5.0f}")
    else:
        print(f"  {'WGC inter-frame interval':<40s}      N/A       N/A    N/A")

    if mss_times:
        avg = statistics.mean(mss_times)
        p95 = sorted(mss_times)[int(len(mss_times) * 0.95)]
        print(f"  {'MSS grab + np.array':<40s}  {avg:7.1f}ms  {p95:7.1f}ms  {1000/avg:5.0f}")

    if wgc_preprocess_times:
        avg = statistics.mean(wgc_preprocess_times)
        p95 = sorted(wgc_preprocess_times)[int(len(wgc_preprocess_times) * 0.95)]
        print(f"  {'WGC frame preprocess (in callback)':<40s}  {avg:7.1f}ms  {p95:7.1f}ms  {1000/avg:5.0f}")

    print()
    print("  Check captures/ folder for saved screenshots.")
    print()


if __name__ == "__main__":
    main()
