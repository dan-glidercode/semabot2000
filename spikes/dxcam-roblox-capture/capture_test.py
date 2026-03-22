"""
DXcam Roblox Capture Spike
===========================
Validates that DXcam can capture frames from a running Roblox window.

Prerequisites:
  - Roblox must be running and visible on screen (joined into a game)
  - Run this script from a terminal

Tests:
  1. Find the Roblox window and get its screen coordinates
  2. Capture frames using DXcam (full screen + cropped to Roblox region)
  3. Save sample screenshots for visual verification
  4. Measure capture latency (single grab and continuous modes)
  5. Validate frames are real content (not black/empty)
  6. Test capture -> OpenCV preprocess cycle with real frames

Run: python capture_test.py
"""

import os
import sys
import time
import statistics
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "captures"
WARMUP = 10
ITERATIONS = 200


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
# 1. Find Roblox window
# ---------------------------------------------------------------------------

def find_roblox_window() -> dict | None:
    """Find the Roblox player window and return its rect."""
    import win32gui

    results = []

    def enum_callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        # Roblox window title is typically "Roblox" or the game name
        title_lower = title.lower()
        if "roblox" in title_lower:
            rect = win32gui.GetWindowRect(hwnd)
            results.append({
                "hwnd": hwnd,
                "title": title,
                "left": rect[0],
                "top": rect[1],
                "right": rect[2],
                "bottom": rect[3],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            })

    win32gui.EnumWindows(enum_callback, None)

    if not results:
        return None

    # Prefer the largest window (the game client, not small popups)
    results.sort(key=lambda w: w["width"] * w["height"], reverse=True)
    return results[0]


# ---------------------------------------------------------------------------
# 2. Frame validation
# ---------------------------------------------------------------------------

def validate_frame(frame: np.ndarray, label: str) -> bool:
    """Check that a frame contains real content, not a black/empty screen."""
    if frame is None:
        print(f"    {label}: FAIL - frame is None")
        return False

    print(f"    {label}: shape={frame.shape}, dtype={frame.dtype}", end="")

    # Check not all black
    mean_val = frame.mean()
    std_val = frame.std()
    print(f", mean={mean_val:.1f}, std={std_val:.1f}", end="")

    if mean_val < 5 and std_val < 2:
        print(" -> LIKELY BLACK/EMPTY")
        return False

    # Check not all same color (frozen/stuck)
    if std_val < 1:
        print(" -> LIKELY FROZEN (uniform color)")
        return False

    print(" -> OK")
    return True


# ---------------------------------------------------------------------------
# 3. Save frame as image
# ---------------------------------------------------------------------------

def save_frame(frame: np.ndarray, name: str):
    """Save a frame as PNG for visual inspection."""
    import cv2
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    # DXcam returns BGR by default when we request it, or RGB otherwise
    cv2.imwrite(str(path), frame)
    size_kb = path.stat().st_size / 1024
    print(f"    Saved: {path} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Main tests
# ---------------------------------------------------------------------------

def main():
    print_header("DXcam Roblox Capture Spike")

    # -- Find Roblox window --
    print_header("1. Finding Roblox Window")
    window = find_roblox_window()
    if window is None:
        print("  ERROR: No Roblox window found!")
        print("  Make sure Roblox is running and you've joined a game.")
        print("  The window title should contain 'Roblox'.")
        sys.exit(1)

    print(f"  Found: \"{window['title']}\"")
    print(f"  Position: ({window['left']}, {window['top']}) -> ({window['right']}, {window['bottom']})")
    print(f"  Size: {window['width']}x{window['height']}")

    # -- Initialize DXcam --
    print_header("2. Initializing DXcam")
    import dxcam

    camera = dxcam.create(output_color="BGR")
    screen_w = camera.width
    screen_h = camera.height
    print(f"  DXcam device created")
    print(f"  Screen resolution: {screen_w}x{screen_h}")

    # DXcam region is (left, top, right, bottom) — clamp to screen bounds
    region = (
        max(0, window["left"]),
        max(0, window["top"]),
        min(screen_w, window["right"]),
        min(screen_h, window["bottom"]),
    )
    print(f"  Capture region (clamped): {region}")

    # -- Test single grab (full screen) --
    print_header("3. Full Screen Grab Test")
    print("  DXcam.grab() returns None if no pixels changed since last call.")
    print("  Retrying up to 20 times with short delays...")
    frame_full = None
    for attempt in range(20):
        frame_full = camera.grab()
        if frame_full is not None:
            break
        time.sleep(0.05)
    if frame_full is not None:
        validate_frame(frame_full, f"Full screen (attempt {attempt + 1})")
        save_frame(frame_full, "01_full_screen")
    else:
        print("  WARNING: Full screen grab returned None after 20 attempts.")
        print("  Trying continuous mode as fallback...")
        try:
            camera.start(target_fps=30)
            time.sleep(0.5)
            frame_full = camera.get_latest_frame()
            camera.stop()
            if frame_full is not None:
                validate_frame(frame_full, "Full screen (continuous)")
                save_frame(frame_full, "01_full_screen")
            else:
                print("  Full screen grab failed in continuous mode too.")
        except Exception as e:
            print(f"  Continuous fallback failed: {e}")

    # -- Test region grab (Roblox window only) --
    print_header("4. Roblox Region Grab Test")
    frame_region = None
    for attempt in range(20):
        frame_region = camera.grab(region=region)
        if frame_region is not None:
            break
        time.sleep(0.05)
    if frame_region is not None:
        validate_frame(frame_region, f"Roblox region (attempt {attempt + 1})")
        save_frame(frame_region, "02_roblox_region")
    else:
        print("  WARNING: Region grab returned None after 20 attempts.")

    # -- Capture latency: single grab mode --
    print_header("5. Capture Latency - grab() Mode")
    print("  (Single frame grabs with small delays between)")

    # Warmup
    for _ in range(WARMUP):
        camera.grab(region=region)
        time.sleep(0.005)

    grab_times = []
    null_count = 0
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        frame = camera.grab(region=region)
        elapsed = (time.perf_counter() - t0) * 1000
        if frame is not None:
            grab_times.append(elapsed)
        else:
            null_count += 1
        time.sleep(0.002)  # small delay so screen has time to change

    print(f"  Successful grabs: {len(grab_times)}/{ITERATIONS} (null: {null_count})")
    if grab_times:
        print_result("grab() with region", grab_times)

    # -- Capture latency: continuous start/get_latest_frame mode --
    print_header("6. Capture Latency - Continuous Mode (start + get_latest_frame)")

    try:
        camera.start(region=region, target_fps=60)
        time.sleep(0.5)  # let the capture thread warm up

        cont_times = []
        null_count = 0
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            frame = camera.get_latest_frame()
            elapsed = (time.perf_counter() - t0) * 1000
            if frame is not None:
                cont_times.append(elapsed)
            else:
                null_count += 1

        camera.stop()

        print(f"  Successful frames: {len(cont_times)}/{ITERATIONS} (null: {null_count})")
        if cont_times:
            print_result("get_latest_frame()", cont_times)
    except Exception as e:
        print(f"  Continuous mode failed: {e}")
        print("  This may be expected on some display configurations.")

    # -- Save multiple sequential frames --
    print_header("7. Sequential Frame Capture (5 frames, 200ms apart)")
    for i in range(5):
        time.sleep(0.2)
        frame = camera.grab(region=region)
        if frame is not None:
            validate_frame(frame, f"Frame {i+1}")
            save_frame(frame, f"03_sequential_{i+1:02d}")
        else:
            print(f"    Frame {i+1}: None (screen unchanged)")

    # -- Capture -> OpenCV preprocess cycle --
    print_header("8. Capture -> Preprocess Cycle (real frames)")
    import cv2

    preprocess_times = []
    for _ in range(WARMUP):
        f = camera.grab(region=region)
        if f is not None:
            cv2.resize(f, (640, 640), interpolation=cv2.INTER_LINEAR)
        time.sleep(0.005)

    for _ in range(min(ITERATIONS, 100)):
        frame = camera.grab(region=region)
        if frame is None:
            time.sleep(0.005)
            continue

        t0 = time.perf_counter()
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        blob *= (1.0 / 255.0)
        blob = blob[np.newaxis]
        elapsed = (time.perf_counter() - t0) * 1000
        preprocess_times.append(elapsed)
        time.sleep(0.005)

    if preprocess_times:
        print(f"  Successful preprocess cycles: {len(preprocess_times)}")
        print_result("Capture(real) -> cv2.resize -> NCHW float32", preprocess_times)
        print(f"  Output shape: {blob.shape}, dtype: {blob.dtype}")
        save_frame(resized, "04_preprocessed_640x640")
    else:
        print("  No frames captured for preprocess test.")

    # -- Cleanup --
    del camera

    # -- Summary --
    print_header("SUMMARY")
    print(f"  Roblox window: \"{window['title']}\" ({window['width']}x{window['height']})")
    if grab_times:
        avg_grab = statistics.mean(grab_times)
        print(f"  grab() avg latency:       {avg_grab:.1f}ms")
    if cont_times:
        avg_cont = statistics.mean(cont_times)
        print(f"  get_latest_frame() avg:   {avg_cont:.1f}ms")
    if preprocess_times:
        avg_pre = statistics.mean(preprocess_times)
        print(f"  Preprocess (real frame):  {avg_pre:.1f}ms")
    print(f"  Screenshots saved to:     {OUTPUT_DIR.resolve()}")
    print()
    print("  Check the captures/ folder to visually confirm the frames")
    print("  contain actual Roblox gameplay (not black or corrupted).")
    print()


if __name__ == "__main__":
    main()
