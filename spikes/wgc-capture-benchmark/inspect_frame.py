"""Quick diagnostic: inspect the WGC Frame object's available methods."""

import threading
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import win32gui

# Find Roblox
results = []
def enum_cb(hwnd, _):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        if title and "roblox" in title.lower():
            results.append(title)
win32gui.EnumWindows(enum_cb, None)

if not results:
    print("No Roblox window found")
    exit(1)

window_name = results[0]
print(f"Targeting: {window_name}")

done = threading.Event()

capture = WindowsCapture(
    cursor_capture=None,
    draw_border=None,
    monitor_index=None,
    window_name=window_name,
)

@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    print(f"\nFrame type: {type(frame)}")
    print(f"Frame dir: {[a for a in dir(frame) if not a.startswith('_')]}")
    # Try common attribute names
    for attr in ["to_numpy", "as_numpy", "numpy", "data", "buffer",
                 "to_ndarray", "raw", "as_raw_nopadding", "save_as_image",
                 "width", "height", "frame_buffer", "as_raw_buffer"]:
        if hasattr(frame, attr):
            val = getattr(frame, attr)
            print(f"  frame.{attr} = {type(val)} {'(callable)' if callable(val) else repr(val)[:100]}")
    capture_control.stop()
    done.set()

@capture.event
def on_closed():
    done.set()

t = threading.Thread(target=capture.start, daemon=True)
t.start()
done.wait(timeout=10)
print("\nDone")
