"""
Screen Capture Fallback Test — MSS (GDI BitBlt) vs DXcam (DXGI)
================================================================
DXcam's DXGI Desktop Duplication failed with COMError on Roblox,
likely due to Byfron anti-cheat blocking DXGI duplication.

This script tests python-mss (GDI BitBlt), which uses a completely
different Windows API path and may bypass the restriction.

Also tests win32gui + PrintWindow as a third alternative.

Run from a regular terminal (not VS Code embedded):
  ..\directml-benchmark\.venv\Scripts\python.exe capture_fallback_test.py
"""

import sys
import time
import statistics
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "captures"
ITERATIONS = 100


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


def save_frame(frame: np.ndarray, name: str):
    import cv2
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    cv2.imwrite(str(path), frame)
    size_kb = path.stat().st_size / 1024
    print(f"    Saved: {path.name} ({size_kb:.0f} KB)")


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
                "hwnd": hwnd,
                "title": title,
                "left": rect[0], "top": rect[1],
                "right": rect[2], "bottom": rect[3],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            })

    win32gui.EnumWindows(enum_cb, None)
    if not results:
        return None
    results.sort(key=lambda w: w["width"] * w["height"], reverse=True)
    return results[0]


# ---------------------------------------------------------------------------
# Method 1: python-mss (GDI BitBlt)
# ---------------------------------------------------------------------------

def test_mss(window: dict):
    print_header("Method 1: python-mss (GDI BitBlt)")

    import mss

    monitor = {
        "left": max(0, window["left"]),
        "top": max(0, window["top"]),
        "width": window["width"],
        "height": window["height"],
    }
    # Clamp width/height to not exceed screen
    print(f"  Capture region: {monitor}")

    try:
        with mss.mss() as sct:
            # Single grab test
            shot = sct.grab(monitor)
            frame = np.array(shot)[:, :, :3]  # BGRA -> BGR
            valid = validate_frame(frame, "Single grab")
            if valid:
                save_frame(frame, "mss_01_single")

            # Latency benchmark
            print("\n  Latency benchmark...")
            times = []
            for _ in range(ITERATIONS):
                t0 = time.perf_counter()
                shot = sct.grab(monitor)
                frame = np.array(shot)[:, :, :3]
                times.append((time.perf_counter() - t0) * 1000)

            print_result("mss.grab() + np.array", times)

            # Save a few sequential frames
            for i in range(3):
                time.sleep(0.2)
                shot = sct.grab(monitor)
                frame = np.array(shot)[:, :, :3]
                validate_frame(frame, f"Sequential {i+1}")
                save_frame(frame, f"mss_02_seq_{i+1:02d}")

            return times

    except Exception as e:
        print(f"  FAILED: {e}")
        return []


# ---------------------------------------------------------------------------
# Method 2: DXcam (DXGI) — expect failure, but test for completeness
# ---------------------------------------------------------------------------

def test_dxcam(window: dict):
    print_header("Method 2: DXcam (DXGI Desktop Duplication)")

    try:
        import dxcam
        camera = dxcam.create(output_color="BGR")
        screen_w = camera.width
        screen_h = camera.height

        region = (
            max(0, window["left"]),
            max(0, window["top"]),
            min(screen_w, window["right"]),
            min(screen_h, window["bottom"]),
        )
        print(f"  Screen: {screen_w}x{screen_h}, region: {region}")

        frame = None
        for attempt in range(10):
            frame = camera.grab(region=region)
            if frame is not None:
                break
            time.sleep(0.05)

        if frame is not None:
            validate_frame(frame, "DXcam grab")
            save_frame(frame, "dxcam_01_single")

            times = []
            for _ in range(ITERATIONS):
                t0 = time.perf_counter()
                f = camera.grab(region=region)
                times.append((time.perf_counter() - t0) * 1000)
                time.sleep(0.002)

            successful = [t for t in times]
            print_result("dxcam.grab()", successful)
            del camera
            return successful
        else:
            print("  All grabs returned None (screen duplication blocked?)")
            del camera
            return []

    except Exception as e:
        print(f"  FAILED: {e}")
        return []


# ---------------------------------------------------------------------------
# Method 3: Win32 PrintWindow
# ---------------------------------------------------------------------------

def test_printwindow(window: dict):
    print_header("Method 3: Win32 PrintWindow")

    try:
        import win32gui
        import win32ui
        import win32con

        hwnd = window["hwnd"]

        # Get client area dimensions
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bottom - top
        print(f"  Client area: {w}x{h}")

        def capture_once():
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
            save_dc.SelectObject(bitmap)
            # PrintWindow with PW_CLIENTONLY flag
            result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
            bmp_info = bitmap.GetInfo()
            bmp_str = bitmap.GetBitmapBits(True)
            frame = np.frombuffer(bmp_str, dtype=np.uint8).reshape(
                bmp_info["bmHeight"], bmp_info["bmWidth"], 4
            )[:, :, :3]  # BGRA -> BGR
            # Cleanup
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            return frame, result

        # Single test
        frame, result = capture_once()
        print(f"  PrintWindow result: {result} (1=success, 0=fail)")
        valid = validate_frame(frame, "PrintWindow")
        if valid:
            save_frame(frame, "printwindow_01_single")

        # Latency benchmark
        print("\n  Latency benchmark...")
        times = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            frame, _ = capture_once()
            times.append((time.perf_counter() - t0) * 1000)

        print_result("PrintWindow capture", times)
        return times

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# Capture -> OpenCV preprocess benchmark (using best working method)
# ---------------------------------------------------------------------------

def test_preprocess_pipeline(capture_func, label: str):
    print_header(f"Preprocess Pipeline: {label}")
    import cv2

    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        frame = capture_func()
        if frame is None:
            continue
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        blob *= (1.0 / 255.0)
        blob = blob[np.newaxis]
        times.append((time.perf_counter() - t0) * 1000)

    if times:
        print_result(f"Capture + preprocess ({label})", times)
        print(f"  Output: shape={blob.shape}, dtype={blob.dtype}")
    else:
        print("  No frames captured.")
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("Screen Capture Fallback Test")

    window = find_roblox_window()
    if window is None:
        print("  ERROR: No Roblox window found!")
        print("  Make sure Roblox is running and you've joined a game.")
        sys.exit(1)

    print(f"  Window: \"{window['title']}\" ({window['width']}x{window['height']})")
    print(f"  HWND: {window['hwnd']}")

    # Test all three methods
    mss_times = test_mss(window)
    dxcam_times = test_dxcam(window)
    pw_times = test_printwindow(window)

    # Preprocess pipeline with best working method
    if mss_times:
        import mss as mss_mod
        monitor = {
            "left": max(0, window["left"]),
            "top": max(0, window["top"]),
            "width": window["width"],
            "height": window["height"],
        }
        sct = mss_mod.mss()

        def mss_capture():
            shot = sct.grab(monitor)
            return np.array(shot)[:, :, :3]

        test_preprocess_pipeline(mss_capture, "MSS")

    # Summary
    print_header("SUMMARY")
    methods = [
        ("python-mss (GDI BitBlt)", mss_times),
        ("DXcam (DXGI Duplication)", dxcam_times),
        ("Win32 PrintWindow", pw_times),
    ]
    print(f"  {'Method':<30s}  {'Avg':>8s}  {'P95':>8s}  {'Status'}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*10}")
    for label, times in methods:
        if times:
            avg = statistics.mean(times)
            p95 = sorted(times)[int(len(times) * 0.95)]
            fps = 1000.0 / avg
            print(f"  {label:<30s}  {avg:7.1f}ms  {p95:7.1f}ms  OK (~{fps:.0f} FPS)")
        else:
            print(f"  {label:<30s}      N/A       N/A  FAILED")

    print()
    print("  Check captures/ folder for saved screenshots.")
    print()


if __name__ == "__main__":
    main()
