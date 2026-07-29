"""
benchmark_pi.py
---------------
Measures inference latency and frames per second of a TFLite model.
Run this ON the Raspberry Pi to produce the edge performance numbers
for the report, then run it on the laptop for comparison.

Usage (on the Pi):
    python3 benchmark_pi.py --model models/mobilenetv2_int8.tflite --runs 100
"""

import argparse
import json
import os
import platform
import time

import numpy as np

from tflite_utils import TFLiteModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a .tflite file")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"[!] Not found: {args.model}")

    model = TFLiteModel(args.model, num_threads=args.threads)
    h, w = model.input_size
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
              for _ in range(args.warmup + args.runs)]

    for i in range(args.warmup):
        model.predict(frames[i])

    times = []
    for i in range(args.warmup, len(frames)):
        t0 = time.perf_counter()
        model.predict(frames[i])
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    result = {
        "device": platform.platform(),
        "processor": platform.machine(),
        "model_file": os.path.basename(args.model),
        "model_size_mb": round(os.path.getsize(args.model) / 1e6, 3),
        "input_size": [h, w],
        "threads": args.threads,
        "runs": int(args.runs),
        "latency_ms_mean": round(float(times.mean()), 2),
        "latency_ms_median": round(float(np.median(times)), 2),
        "latency_ms_p95": round(float(np.percentile(times, 95)), 2),
        "latency_ms_min": round(float(times.min()), 2),
        "latency_ms_max": round(float(times.max()), 2),
        "fps_mean": round(1000.0 / float(times.mean()), 2),
    }

    print("\n===== inference benchmark =====")
    for k, v in result.items():
        print(f"  {k:22s}: {v}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[i] saved {args.out}")
    print("[i] Screenshot this terminal output for the report appendix.")


if __name__ == "__main__":
    main()
