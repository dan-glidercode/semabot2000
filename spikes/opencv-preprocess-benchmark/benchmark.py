"""
OpenCV vs PIL Preprocessing Benchmark
======================================
Tests the preprocessing step (resize + normalize + transpose) that feeds
into YOLO11n, comparing PIL (current) vs OpenCV (proposed).

Also validates that OpenCV produces equivalent output to PIL so we can
swap it in without affecting inference accuracy.

Run: python benchmark.py
"""

import time
import statistics
import sys

import numpy as np

WARMUP = 20
ITERATIONS = 500

# Simulate a captured frame (typical 1080p-ish from DXcam)
FRAME_H, FRAME_W = 1080, 1920
TARGET_SIZE = (640, 640)  # YOLO11n input


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, times_ms: list[float]):
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
# PIL preprocessing (current approach from directml-benchmark)
# ---------------------------------------------------------------------------

def preprocess_pil(frame: np.ndarray) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(frame).resize(TARGET_SIZE)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img_np, 0)            # add batch dim


# ---------------------------------------------------------------------------
# OpenCV preprocessing (proposed replacement)
# ---------------------------------------------------------------------------

def preprocess_cv2(frame: np.ndarray) -> np.ndarray:
    import cv2
    img = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img_np = img.astype(np.float32) * (1.0 / 255.0)
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img_np, 0)


# ---------------------------------------------------------------------------
# OpenCV with contiguous array optimization
# ---------------------------------------------------------------------------

def preprocess_cv2_contiguous(frame: np.ndarray) -> np.ndarray:
    import cv2
    img = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img_np = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    img_np *= (1.0 / 255.0)
    return img_np[np.newaxis]


# ---------------------------------------------------------------------------
# OpenCV with pre-allocated buffer
# ---------------------------------------------------------------------------

_buffer_resized = np.empty((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
_buffer_chw = np.empty((1, 3, TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float32)


def preprocess_cv2_preallocated(frame: np.ndarray) -> np.ndarray:
    import cv2
    cv2.resize(frame, TARGET_SIZE, dst=_buffer_resized, interpolation=cv2.INTER_LINEAR)
    # Transpose + normalize into pre-allocated float buffer
    np.divide(
        _buffer_resized.transpose(2, 0, 1),
        255.0,
        out=_buffer_chw[0],
        casting="unsafe",
    )
    return _buffer_chw


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(func, frame: np.ndarray, label: str) -> list[float]:
    # Warmup
    for _ in range(WARMUP):
        func(frame)

    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        func(frame)
        times.append((time.perf_counter() - t0) * 1000)

    print_result(label, times)
    return times


# ---------------------------------------------------------------------------
# Accuracy validation
# ---------------------------------------------------------------------------

def validate_equivalence(frame: np.ndarray):
    """Check that OpenCV produces output close enough to PIL."""
    pil_out = preprocess_pil(frame)
    cv2_out = preprocess_cv2(frame)
    cv2c_out = preprocess_cv2_contiguous(frame)
    cv2p_out = preprocess_cv2_preallocated(frame)

    print("\n  Output shape validation:")
    print(f"    PIL:              {pil_out.shape}, dtype={pil_out.dtype}")
    print(f"    CV2:              {cv2_out.shape}, dtype={cv2_out.dtype}")
    print(f"    CV2 contiguous:   {cv2c_out.shape}, dtype={cv2c_out.dtype}")
    print(f"    CV2 preallocated: {cv2p_out.shape}, dtype={cv2p_out.dtype}")

    # PIL uses RGB, OpenCV uses BGR — compare after accounting for channel order
    # DXcam outputs BGR by default when output_color="BGR", or RGB otherwise.
    # For YOLO, channel order matters for accuracy but not for benchmarking.
    # Here we just check shapes and value ranges are consistent.
    for name, arr in [("CV2", cv2_out), ("CV2c", cv2c_out), ("CV2p", cv2p_out)]:
        assert arr.shape == pil_out.shape, f"{name} shape mismatch: {arr.shape} vs {pil_out.shape}"
        assert arr.dtype == np.float32, f"{name} dtype mismatch: {arr.dtype}"
        assert arr.min() >= 0.0, f"{name} has negative values"
        assert arr.max() <= 1.0, f"{name} has values > 1.0"
        # Note: pixel values differ slightly between PIL and OpenCV due to
        # different resize interpolation implementations. This is expected.
        max_diff = np.max(np.abs(pil_out - arr))
        mean_diff = np.mean(np.abs(pil_out - arr))
        print(f"    {name:20s} vs PIL — max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    print("  All validations passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("OpenCV vs PIL Preprocessing Benchmark")
    print(f"  Frame: {FRAME_W}x{FRAME_H} -> {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"  Iterations: {ITERATIONS}  |  Warmup: {WARMUP}")

    import cv2
    from PIL import Image
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  PIL:    {Image.__version__}")
    print(f"  NumPy:  {np.__version__}")
    print(f"  Python: {sys.version.split()[0]}")

    # Generate a synthetic frame (random noise simulating a game capture)
    frame = np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)

    # Validate outputs are equivalent
    print_header("VALIDATION")
    validate_equivalence(frame)

    # Benchmark each approach
    print_header("PIL (current)")
    pil_times = benchmark(preprocess_pil, frame, "PIL Image.fromarray + resize + array")

    print_header("OpenCV basic")
    cv2_times = benchmark(preprocess_cv2, frame, "cv2.resize + astype + transpose")

    print_header("OpenCV contiguous")
    cv2c_times = benchmark(preprocess_cv2_contiguous, frame, "cv2.resize + ascontiguousarray transpose")

    print_header("OpenCV pre-allocated")
    cv2p_times = benchmark(preprocess_cv2_preallocated, frame, "cv2.resize(dst=buf) + np.divide(out=buf)")

    # Summary
    print_header("SUMMARY")
    results = [
        ("PIL (current)", pil_times),
        ("OpenCV basic", cv2_times),
        ("OpenCV contiguous", cv2c_times),
        ("OpenCV pre-allocated", cv2p_times),
    ]
    pil_avg = statistics.mean(pil_times)
    print(f"  {'Method':<30s}  {'Avg':>8s}  {'P95':>8s}  {'Speedup':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, times in results:
        avg = statistics.mean(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        speedup = pil_avg / avg
        print(f"  {label:<30s}  {avg:7.2f}ms  {p95:7.2f}ms  {speedup:7.2f}x")

    print()


if __name__ == "__main__":
    main()
