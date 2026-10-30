"""Drift check for long-file separation.

Verifies that the separated output stays time-aligned with the source audio
across the whole file, in particular beyond the 20 minute mark.

Two independent checks are performed per checkpoint position:

  1. Cross-correlation lag between each stem and the source mix. A drifting
     implementation shows a non-zero lag that grows with time.
  2. Energy reconstruction: the sum of the four stems must approximate the
     source mix (the masks sum to one by construction).

The source mix is decoded once with ffmpeg.
"""

import argparse
import json
import os
import subprocess

import numpy as np

SAMPLE_RATE = 44100


def decode(path, sample_rate=SAMPLE_RATE):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2",
         "-ar", str(sample_rate), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)


def best_lag(a, b, max_lag):
    """Lag (in samples) that best aligns `b` onto `a`."""
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    if n < max_lag * 4 + 1024:
        return 0
    # Compare the interior region only, so every candidate lag has the same
    # number of overlapping samples.
    lo = max_lag
    hi = n - max_lag
    if hi - lo < 1024:
        return 0
    best = 0
    best_score = -1e30
    for lag in range(-max_lag, max_lag + 1, 16):
        if lag < 0:
            score = float(np.dot(b[lo - lag:hi - lag], a[lo:hi]))
        else:
            score = float(np.dot(b[lo:hi - lag], a[lo + lag:hi]))
        if score > best_score:
            best_score = score
            best = lag
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp_dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--positions", default="60,300,600,900,1200,1500,1800,2100,2400")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    print("Decoding source mix...")
    mix = decode(args.source)
    print(f"source samples: {mix.shape[0]} ({mix.shape[0] / SAMPLE_RATE:.1f}s)")

    stems = {}
    for name in ("vocals", "drums", "bass", "other"):
        path = os.path.join(args.cpp_dir, f"{name}.wav")
        if os.path.exists(path):
            stems[name] = decode(path)
    print(f"stems decoded: {sorted(stems)}")
    for name, data in stems.items():
        print(f"  {name}: {data.shape[0]} samples "
              f"({data.shape[0] / SAMPLE_RATE:.1f}s)")

    report = {"source_samples": int(mix.shape[0]), "stems": {}, "checks": []}

    # Length consistency.
    for name, data in stems.items():
        report["stems"][name] = {
            "samples": int(data.shape[0]),
            "seconds": round(data.shape[0] / SAMPLE_RATE, 3),
            "matches_source": bool(data.shape[0] == mix.shape[0]),
        }

    max_lag = 44100  # one second
    for position in [int(p) for p in args.positions.split(",")]:
        start = position * SAMPLE_RATE
        window = 30 * SAMPLE_RATE
        end = min(start + window, mix.shape[0])
        if start >= mix.shape[0]:
            continue
        entry = {"position_sec": position, "lags": {}}
        for name, data in stems.items():
            if start >= data.shape[0]:
                continue
            stop = min(end, data.shape[0])
            ref = mix[start:stop, 0]
            cand = data[start:stop, 0]
            lag = best_lag(ref, cand, max_lag)
            entry["lags"][name] = lag

        # Reconstruction: sum of the stems versus the mix.
        stack = np.stack([stems[n][start:end] for n in ("vocals", "drums", "bass", "other")
                          if n in stems and start < stems[n].shape[0]])
        if stack.shape[0] == 4:
            recon = stack.sum(axis=0)
            ref = mix[start:end]
            err = float(np.sqrt(np.mean((recon - ref).astype(np.float64) ** 2)))
            ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
            entry["reconstruction_snr_db"] = round(
                20 * np.log10(ref_rms / err) if err > 0 else float("inf"), 2)

        report["checks"].append(entry)
        lag_text = ", ".join(f"{k}={v}" for k, v in entry["lags"].items())
        print(f"t={position:>5}s  lags: {lag_text}  "
              f"recon_snr={entry.get('reconstruction_snr_db')}dB")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
