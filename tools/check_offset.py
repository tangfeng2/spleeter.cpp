"""Confirm the constant sample offset is inherent to Spleeter, not a port bug.

Compares, at several positions across the file, the cross-correlation lag of
each stem against the source mix for:
  * the Python Spleeter reference (20 minute region), and
  * the C++ port output (full file).

A constant offset in both means the port reproduces Spleeter exactly; a lag
that grows with time would mean the port drifts.
"""

import argparse
import os
import subprocess

import numpy as np

SAMPLE_RATE = 44100


def decode(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)


def best_lag(a, b, max_lag):
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    lo, hi = max_lag, n - max_lag
    if hi - lo < 1024:
        return 0
    best, best_score = 0, -1e30
    for lag in range(-max_lag, max_lag + 1, 4):
        if lag < 0:
            score = float(np.dot(b[lo - lag:hi - lag], a[lo:hi]))
        else:
            score = float(np.dot(b[lo:hi - lag], a[lo + lag:hi]))
        if score > best_score:
            best_score, best = score, lag
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cpp_dir", required=True)
    parser.add_argument("--cpp_offset", type=int, default=0)
    parser.add_argument("--py_dir", default="")
    parser.add_argument("--py_offset", type=int, default=0)
    parser.add_argument("--positions", default="0,60,300,600,900,1200,1500,1800,2100,2400")
    args = parser.parse_args()

    mix = decode(args.source)
    window = 30 * SAMPLE_RATE
    max_lag = 22050

    for label, directory, offset in (
        ("C++ port", args.cpp_dir, args.cpp_offset),
        ("Python Spleeter", args.py_dir, args.py_offset),
    ):
        if not directory:
            continue
        print(f"=== {label} ===")
        for position in [int(p) for p in args.positions.split(",")]:
            start = position * SAMPLE_RATE
            if start >= mix.shape[0]:
                continue
            end = min(start + window, mix.shape[0])
            lags = {}
            for name in ("vocals", "drums", "bass", "other"):
                if name == "vocals" or True:
                    path = os.path.join(directory, f"{name}.wav")
                    if not os.path.exists(path):
                        path = os.path.join(directory, f"{name}.npy")
                    if not os.path.exists(path):
                        continue
                    data = decode(path) if path.endswith(".wav") else np.load(path)
                    # Position of this window inside the stem's own timeline.
                    stem_start = start - offset * SAMPLE_RATE
                    if stem_start < 0:
                        continue
                    ref = mix[start:end, 0]
                    cand = data[stem_start:stem_start + (end - start), 0]
                    lags[name] = best_lag(ref, cand, max_lag)
            print(f"  t={position:>5}s  " +
                  "  ".join(f"{k}={v:+d}" for k, v in lags.items()))


if __name__ == "__main__":
    main()
