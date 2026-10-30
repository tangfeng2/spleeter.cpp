"""Definitive drift test over a long file.

For many positions across the file, the alignment between the separated stems
and the source audio is measured at 1-sample resolution. A drifting
implementation would show the measured lag growing with time; a correct one
shows a constant lag.

The same measurement is applied to the Python Spleeter reference for the
same absolute positions, so the two can be compared directly.
"""

import argparse
import json
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
    """Lag at 1-sample resolution maximizing the correlation."""
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    lo, hi = max_lag, n - max_lag
    if hi - lo < 2048:
        return None, 0.0
    # Correlation for lag 0 plus the local slope, refined around the peak.
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            scores.append(float(np.dot(b[lo - lag:hi - lag], a[lo:hi])))
        else:
            scores.append(float(np.dot(b[lo:hi - lag], a[lo + lag:hi])))
    scores = np.asarray(scores)
    peak = int(np.argmax(scores)) - max_lag
    # Sub-sample parabolic refinement.
    if 0 < peak + max_lag < len(scores) - 1:
        y0, y1, y2 = scores[peak + max_lag - 1:peak + max_lag + 2]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            peak += 0.5 * (y0 - y2) / denom
    return peak, float(scores.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cpp_dir", required=True)
    parser.add_argument("--py_dir", default="")
    parser.add_argument("--py_offset", type=int, default=0)
    parser.add_argument("--positions", default="30,300,600,900,1200,1500,1800,2100,2400")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    mix = decode(args.source)
    window = 20 * SAMPLE_RATE
    max_lag = 400

    stems = {}
    for name in ("vocals", "drums", "bass", "other"):
        path = os.path.join(args.cpp_dir, f"{name}.wav")
        if os.path.exists(path):
            stems[name] = decode(path)

    report = {"cpp": [], "python": []}

    for label, directory, offset, key in (
        ("cpp", args.cpp_dir, 0, "cpp"),
        ("python", args.py_dir, args.py_offset, "python"),
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
                path = os.path.join(directory, f"{name}.wav")
                if not os.path.exists(path):
                    path = os.path.join(directory, f"{name}.npy")
                if not os.path.exists(path):
                    continue
                data = decode(path) if path.endswith(".wav") else np.load(path)
                stem_start = start - offset * SAMPLE_RATE
                if stem_start < 0:
                    continue
                ref = mix[start:end, 0]
                cand = data[stem_start:stem_start + (end - start), 0]
                lag, score = best_lag(ref, cand, max_lag)
                if lag is not None:
                    lags[name] = round(lag, 3)
            if not lags:
                continue
            report[key].append({"position_sec": position, "lags": lags})
            print(f"  t={position:>5}s  " +
                  "  ".join(f"{k}={v:+g}" for k, v in lags.items()))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
