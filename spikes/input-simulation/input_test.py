"""
Input Simulation Spike — Roblox Compatibility Test
====================================================
Tests whether various input simulation methods are accepted by Roblox
or blocked by Byfron/Hyperion anti-cheat.

Methods tested:
  1. pynput (uses SendInput internally)
  2. pydirectinput (DirectInput scan codes via SendInput)
  3. pyautogui (uses keybd_event / mouse_event — deprecated APIs)
  4. ctypes SendInput directly (low-level control)

Each method will:
  - Press W for 0.5s (walk forward)
  - Press Space (jump)
  - Move mouse slightly

The user must observe Roblox to see if the character responds.

IMPORTANT: Roblox must be the focused foreground window when input is sent.

Run: python input_test.py
"""

import sys
import time
import ctypes
import ctypes.wintypes
from pathlib import Path

# ---------------------------------------------------------------------------
# Win32 constants for direct ctypes SendInput
# ---------------------------------------------------------------------------

INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

# Scan codes (DirectInput)
SCAN_W = 0x11
SCAN_A = 0x1E
SCAN_S = 0x1F
SCAN_D = 0x20
SCAN_SPACE = 0x39

# Structures for SendInput
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.wintypes.DWORD), ("union", INPUT_UNION)]


def send_input_raw(*inputs):
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    ctypes.windll.user32.SendInput(n, ctypes.pointer(arr), ctypes.sizeof(INPUT))


def make_key_input(scan_code, flags=KEYEVENTF_SCANCODE):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wScan = scan_code
    inp.union.ki.dwFlags = flags
    return inp


def make_mouse_move(dx, dy):
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = dx
    inp.union.mi.dy = dy
    inp.union.mi.dwFlags = MOUSEEVENTF_MOVE
    return inp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def find_roblox_window():
    import win32gui
    results = []

    def enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and "roblox" in title.lower():
                results.append({"hwnd": hwnd, "title": title})

    win32gui.EnumWindows(enum_cb, None)
    return results[0] if results else None


def focus_roblox(hwnd):
    """Bring Roblox to foreground."""
    import win32gui
    import win32con
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"    Could not focus Roblox: {e}")
        return False


def countdown(seconds: int, message: str):
    print(f"\n  {message}")
    for i in range(seconds, 0, -1):
        print(f"    {i}...", flush=True)
        time.sleep(1)
    print("    GO!")


# ---------------------------------------------------------------------------
# Test 1: pynput
# ---------------------------------------------------------------------------

def test_pynput():
    print_header("Method 1: pynput")
    print("  Uses SendInput internally via ctypes.")

    try:
        from pynput.keyboard import Controller as KbCtrl, Key
        from pynput.mouse import Controller as MouseCtrl

        kb = KbCtrl()
        mouse = MouseCtrl()

        # Keyboard: press W for 0.5s
        print("  [keyboard] Pressing W for 0.5s (walk forward)...")
        t0 = time.perf_counter()
        kb.press('w')
        time.sleep(0.5)
        kb.release('w')
        kb_time = (time.perf_counter() - t0) * 1000
        print(f"    Key press/release round-trip: {kb_time:.1f}ms")

        time.sleep(0.3)

        # Keyboard: press Space (jump)
        print("  [keyboard] Pressing Space (jump)...")
        t0 = time.perf_counter()
        kb.press(Key.space)
        time.sleep(0.1)
        kb.release(Key.space)
        space_time = (time.perf_counter() - t0) * 1000
        print(f"    Space press/release: {space_time:.1f}ms")

        time.sleep(0.3)

        # Mouse: move slightly
        print("  [mouse] Moving mouse 50px right, 50px down...")
        t0 = time.perf_counter()
        mouse.move(50, 50)
        mouse_time = (time.perf_counter() - t0) * 1000
        print(f"    Mouse move: {mouse_time:.2f}ms")

        time.sleep(0.3)

        # Latency benchmark: rapid key presses
        print("  [benchmark] 50 rapid key taps (W)...")
        tap_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            kb.press('w')
            kb.release('w')
            tap_times.append((time.perf_counter() - t0) * 1000)
            time.sleep(0.01)

        avg_tap = sum(tap_times) / len(tap_times)
        print(f"    Avg tap latency: {avg_tap:.2f}ms")

        return True, avg_tap

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, 0


# ---------------------------------------------------------------------------
# Test 2: pydirectinput
# ---------------------------------------------------------------------------

def test_pydirectinput():
    print_header("Method 2: pydirectinput (DirectInput scan codes)")
    print("  Uses SendInput with DirectInput scan codes.")
    print("  Designed specifically for game input.")

    try:
        import pydirectinput

        # Disable pause between actions for benchmarking
        pydirectinput.PAUSE = 0

        # Keyboard: press W for 0.5s
        print("  [keyboard] Pressing W for 0.5s (walk forward)...")
        t0 = time.perf_counter()
        pydirectinput.keyDown('w')
        time.sleep(0.5)
        pydirectinput.keyUp('w')
        kb_time = (time.perf_counter() - t0) * 1000
        print(f"    Key press/release round-trip: {kb_time:.1f}ms")

        time.sleep(0.3)

        # Keyboard: press Space (jump)
        print("  [keyboard] Pressing Space (jump)...")
        t0 = time.perf_counter()
        pydirectinput.keyDown('space')
        time.sleep(0.1)
        pydirectinput.keyUp('space')
        space_time = (time.perf_counter() - t0) * 1000
        print(f"    Space press/release: {space_time:.1f}ms")

        time.sleep(0.3)

        # Mouse: move slightly
        print("  [mouse] Moving mouse 50px right, 50px down...")
        t0 = time.perf_counter()
        pydirectinput.moveRel(50, 50)
        mouse_time = (time.perf_counter() - t0) * 1000
        print(f"    Mouse move: {mouse_time:.2f}ms")

        time.sleep(0.3)

        # Latency benchmark
        print("  [benchmark] 50 rapid key taps (W)...")
        tap_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            pydirectinput.keyDown('w')
            pydirectinput.keyUp('w')
            tap_times.append((time.perf_counter() - t0) * 1000)
            time.sleep(0.01)

        avg_tap = sum(tap_times) / len(tap_times)
        print(f"    Avg tap latency: {avg_tap:.2f}ms")

        return True, avg_tap

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, 0


# ---------------------------------------------------------------------------
# Test 3: pyautogui
# ---------------------------------------------------------------------------

def test_pyautogui():
    print_header("Method 3: pyautogui (keybd_event / mouse_event)")
    print("  Uses deprecated Win32 APIs. May not work in games.")

    try:
        import pyautogui

        pyautogui.PAUSE = 0

        # Keyboard: press W for 0.5s
        print("  [keyboard] Pressing W for 0.5s (walk forward)...")
        t0 = time.perf_counter()
        pyautogui.keyDown('w')
        time.sleep(0.5)
        pyautogui.keyUp('w')
        kb_time = (time.perf_counter() - t0) * 1000
        print(f"    Key press/release round-trip: {kb_time:.1f}ms")

        time.sleep(0.3)

        # Keyboard: press Space (jump)
        print("  [keyboard] Pressing Space (jump)...")
        t0 = time.perf_counter()
        pyautogui.keyDown('space')
        time.sleep(0.1)
        pyautogui.keyUp('space')
        space_time = (time.perf_counter() - t0) * 1000
        print(f"    Space press/release: {space_time:.1f}ms")

        time.sleep(0.3)

        # Mouse: move slightly
        print("  [mouse] Moving mouse 50px right, 50px down...")
        t0 = time.perf_counter()
        pyautogui.moveRel(50, 50)
        mouse_time = (time.perf_counter() - t0) * 1000
        print(f"    Mouse move: {mouse_time:.2f}ms")

        time.sleep(0.3)

        # Latency benchmark
        print("  [benchmark] 50 rapid key taps (W)...")
        tap_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            pyautogui.keyDown('w')
            pyautogui.keyUp('w')
            tap_times.append((time.perf_counter() - t0) * 1000)
            time.sleep(0.01)

        avg_tap = sum(tap_times) / len(tap_times)
        print(f"    Avg tap latency: {avg_tap:.2f}ms")

        return True, avg_tap

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, 0


# ---------------------------------------------------------------------------
# Test 4: ctypes SendInput directly
# ---------------------------------------------------------------------------

def test_ctypes_sendinput():
    print_header("Method 4: ctypes SendInput (direct, scan codes)")
    print("  Raw Win32 SendInput with DirectInput scan codes.")
    print("  Lowest-level Python approach without a driver.")

    try:
        # Keyboard: press W for 0.5s
        print("  [keyboard] Pressing W for 0.5s (walk forward)...")
        t0 = time.perf_counter()
        send_input_raw(make_key_input(SCAN_W))
        time.sleep(0.5)
        send_input_raw(make_key_input(SCAN_W, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP))
        kb_time = (time.perf_counter() - t0) * 1000
        print(f"    Key press/release round-trip: {kb_time:.1f}ms")

        time.sleep(0.3)

        # Keyboard: press Space (jump)
        print("  [keyboard] Pressing Space (jump)...")
        t0 = time.perf_counter()
        send_input_raw(make_key_input(SCAN_SPACE))
        time.sleep(0.1)
        send_input_raw(make_key_input(SCAN_SPACE, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP))
        space_time = (time.perf_counter() - t0) * 1000
        print(f"    Space press/release: {space_time:.1f}ms")

        time.sleep(0.3)

        # Mouse: move
        print("  [mouse] Moving mouse 50px right, 50px down...")
        t0 = time.perf_counter()
        send_input_raw(make_mouse_move(50, 50))
        mouse_time = (time.perf_counter() - t0) * 1000
        print(f"    Mouse move: {mouse_time:.2f}ms")

        time.sleep(0.3)

        # Latency benchmark
        print("  [benchmark] 50 rapid key taps (W)...")
        tap_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            send_input_raw(make_key_input(SCAN_W))
            send_input_raw(make_key_input(SCAN_W, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP))
            tap_times.append((time.perf_counter() - t0) * 1000)
            time.sleep(0.01)

        avg_tap = sum(tap_times) / len(tap_times)
        print(f"    Avg tap latency: {avg_tap:.2f}ms")

        return True, avg_tap

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("Input Simulation Spike - Roblox Compatibility")

    window = find_roblox_window()
    if window is None:
        print("  ERROR: No Roblox window found!")
        print("  Start Roblox and join a game first.")
        sys.exit(1)

    print(f"  Found: \"{window['title']}\"")

    print("\n  INSTRUCTIONS:")
    print("  - Roblox must be the FOCUSED window when each test runs")
    print("  - Watch your character in-game to see if it responds")
    print("  - Each test will: press W (walk), Space (jump), move mouse")
    print("  - After all tests, you'll report which methods worked")

    # Focus Roblox
    print("\n  Attempting to focus Roblox window...")
    focus_roblox(window["hwnd"])

    countdown(5, "Starting tests in 5 seconds — switch to Roblox NOW!")

    results = {}

    # Run each test with a pause between
    for name, func in [
        ("pynput", test_pynput),
        ("pydirectinput", test_pydirectinput),
        ("pyautogui", test_pyautogui),
        ("ctypes SendInput", test_ctypes_sendinput),
    ]:
        success, latency = func()
        results[name] = {"success": success, "latency_ms": latency}
        if success:
            print(f"\n  >> Did the character respond to {name}? (observe in-game)")
        time.sleep(1.5)

    # Summary
    print_header("SUMMARY")
    print(f"  {'Method':<25s}  {'API Calls OK':>12s}  {'Avg Tap Latency':>16s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*16}")
    for name, r in results.items():
        status = "YES" if r["success"] else "FAILED"
        lat = f"{r['latency_ms']:.2f}ms" if r["success"] else "N/A"
        print(f"  {name:<25s}  {status:>12s}  {lat:>16s}")

    print()
    print("  NOTE: 'API Calls OK' means the Python call succeeded without error.")
    print("  Whether Roblox ACCEPTED the input depends on what you observed.")
    print("  Please note which methods made your character move/jump/look around.")
    print()


if __name__ == "__main__":
    main()
